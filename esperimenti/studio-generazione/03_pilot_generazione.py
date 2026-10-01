"""Passo 5 di M2: pilot di generazione su 5 domande, un generatore, giudizio a mano.

Salta deliberatamente il passo locale (misura di velocita' di generatori scaricati via
Ollama/llama-cpp-python): il costo di una campagna via API e' di centesimi, il tempo per
installare, scaricare e misurare quattro modelli locali su questa CPU non lo sarebbe, e la
sola cosa che quella misura avrebbe insegnato (quale gira un po' piu' in fretta qui) non e'
una competenza trasferibile. Decisione registrata nel banco, voce "Generatore".

Modello scelto: Gemini 3.1 Pro (id API "gemini-3.1-pro-preview"), non la variante piu'
economica (Flash-Lite): il pilot serve a giudicare i modi di fallire della generazione, e un
modello debole introdurrebbe un fattore di confusione (non capisce l'istruzione vs. il
contesto non basta) proprio dove serve isolare l'effetto del contesto. Il costo di 5 domande
a questo prezzo e' comunque frazioni di centesimo.

Le cinque domande, scelte per regime di recupero (vedi Appunti1.md):
  A  - oro al primo posto (la stessa di 01_lettura_manuale.py)
  B  - oro fra i candidati ma fuori dai primi dieci (idem)
  C1 - nessun oro fra i 60 candidati (idem)
  D  - oro fra i primi dieci ma non al primo posto (nuova: regime maggioritario, 72/180,
       non ancora rappresentato nel pilot)
  C2 - un secondo caso di regime C, da una conversazione diversa da C1: le tre domande di
       01_lettura_manuale.py erano tutte della stessa conversazione per coincidenza
       dell'ordinamento, e un secondo esempio scorrelato serve a non generalizzare da un
       singolo dialogo.

Il testo passato al generatore e' la domanda "rewrite" (riscritta a mano dall'annotatore):
e' quella su cui il recupero e' stato eseguito, quindi l'unica coerente con il contesto
recuperato. La forma "lastturn" (quello che l'utente scrive davvero) resta in output solo
come riferimento per il giudizio manuale.

Produce:
  output/pilot_generazione.json  - dati strutturati (domanda, contesto, risposta, metadati)
  output/pilot_generazione.md    - stessa cosa, leggibile

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-generazione/03_pilot_generazione.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
M1_OUT = REPO_ROOT / "esperimenti" / "studio-embedding-retrieval" / "output"
OUT_DIR = Path(__file__).resolve().parent / "output"
ENV_PATH = Path(__file__).resolve().parent / "apikey.env"

RERANK_DEPTH = 30
K = 10
MODEL = "gemini-3.1-pro-preview"

PROMPT_TEMPLATE = """You are a financial assistant answering questions from an online forum. \
Use ONLY the numbered passages below to answer. Cite the passages you rely on by their \
number in square brackets, e.g. [3], for every factual claim. If the passages do not contain \
enough information to answer the question, say so explicitly instead of guessing.

Passages:
{passages}

Question: {question}

Answer:"""


def load_corpus(path: Path):
    ids, titles, texts = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["_id"])
            titles.append(r.get("title", ""))
            texts.append(r["text"])
    return ids, titles, texts


def load_qrels(path: Path):
    gold = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            q, c, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                gold.setdefault(q, set()).add(c)
    return gold


def load_variant(path: Path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["_id"]] = r["text"].replace("|user|:", " ").strip()
    return out


def conv_id(query_id: str) -> str:
    return query_id.split("<::>")[0]


def main() -> None:
    load_dotenv(ENV_PATH)
    api_key = os.environ["GEMINI_API"]
    client = genai.Client(api_key=api_key)

    ids, titles, texts = load_corpus(CORPUS_PATH)
    id_to_idx = {c: i for i, c in enumerate(ids)}
    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    rewrite = load_variant(MTRAG / "fiqa_rewrite.jsonl")
    lastturn = load_variant(MTRAG / "fiqa_lastturn.jsonl")

    d = np.load(M1_OUT / "configurazione_finale_candidati.npz", allow_pickle=True)
    cand = d["cand"]
    sc = np.load(M1_OUT / "configurazione_finale_scores.npy")

    finale = []
    for i in range(len(query_ids)):
        testa = list(np.argsort(-sc[i][:RERANK_DEPTH], kind="stable"))
        coda = list(range(RERANK_DEPTH, cand.shape[1]))
        finale.append([int(cand[i][p]) for p in testa + coda])

    regimi = {"A": [], "B": [], "C": [], "D": []}
    for i, q in enumerate(query_ids):
        gi = {id_to_idx[g] for g in gold_by_query[q] if g in id_to_idx}
        pos = [finale[i].index(g) + 1 for g in gi if g in finale[i]]
        if pos and min(pos) == 1:
            regimi["A"].append(i)
        elif pos and 1 < min(pos) <= K:
            regimi["D"].append(i)
        elif pos and min(pos) > K:
            regimi["B"].append(i)
        elif not pos:
            regimi["C"].append(i)

    scelte = {
        "A": regimi["A"][0],
        "B": regimi["B"][0],
        "C1": regimi["C"][0],
        "D": regimi["D"][0],
    }
    conv_c1 = conv_id(query_ids[scelte["C1"]])
    scelte["C2"] = next(i for i in regimi["C"][1:] if conv_id(query_ids[i]) != conv_c1)

    etichette = {
        "A": "oro al primo posto",
        "B": "oro fra i candidati ma fuori dai primi dieci",
        "C1": "nessun oro fra i 60 candidati",
        "D": "oro fra i primi dieci ma non al primo posto",
        "C2": "nessun oro fra i 60 candidati, seconda conversazione",
    }

    risultati = []
    for key, i in scelte.items():
        q = query_ids[i]
        gi = {id_to_idx[g] for g in gold_by_query[q] if g in id_to_idx}
        pos = {ids[g]: (finale[i].index(g) + 1 if g in finale[i] else None) for g in gi}
        passaggi = [ids[g] for g in finale[i][:K]]
        blocco_passaggi = "\n\n".join(
            f"[{n}] {texts[id_to_idx[pid]]}" for n, pid in enumerate(passaggi, start=1)
        )
        domanda = rewrite[q]
        prompt = PROMPT_TEMPLATE.format(passages=blocco_passaggi, question=domanda)

        for tentativo in range(3):
            try:
                resp = client.models.generate_content(model=MODEL, contents=prompt)
                risposta = resp.text
                break
            except Exception as e:  # rate limit o errore transitorio
                if tentativo == 2:
                    risposta = f"[ERRORE dopo 3 tentativi: {e}]"
                else:
                    time.sleep(5)

        risultati.append(
            {
                "regime": key,
                "etichetta": etichette[key],
                "query_id": q,
                "conversazione": conv_id(q),
                "domanda_ultimo_turno": lastturn.get(q, "—"),
                "domanda_riscritta": domanda,
                "fonti_annotate": {pid: p for pid, p in pos.items()},
                "passaggi_nel_contesto": passaggi,
                "risposta": risposta,
            }
        )
        print(f"regime {key}: generato ({len(risposta)} caratteri)")

    (OUT_DIR / "pilot_generazione.json").write_text(
        json.dumps(risultati, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    righe = ["# Pilot di generazione — 5 domande, un regime ciascuna", "",
             f"Modello: `{MODEL}`. Domanda passata al generatore: forma riscritta (coerente",
             "col recupero). Il giudizio a mano si fa nella pagina di revisione, non qui.", ""]
    for r in risultati:
        righe += [
            "---", "",
            f"## Regime {r['regime']} — {r['etichetta']}",
            "",
            f"**Domanda riscritta (passata al generatore)**: {r['domanda_riscritta']}",
            "",
            f"**Ultimo turno reale**: {r['domanda_ultimo_turno']}",
            "",
            f"**Fonti annotate e posizione finale**: {r['fonti_annotate'] or 'nessuna'}",
            "",
            "**Risposta generata**:",
            "",
            r["risposta"],
            "",
        ]
    (OUT_DIR / "pilot_generazione.md").write_text("\n".join(righe), encoding="utf-8")
    print("scritto: output/pilot_generazione.json, output/pilot_generazione.md")


if __name__ == "__main__":
    main()
