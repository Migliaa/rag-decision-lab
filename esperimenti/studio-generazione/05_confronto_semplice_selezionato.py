"""Passo 6, seconda parte: generare su un piccolo taglio delle domande dove i contesti
differiscono, non su tutte le 64 (Andrea: e' un esperimento per imparare, non per dimostrare
un risultato, un campione piu' piccolo va bene se dichiarato esplorativo).

Sei domande, tre dove il recupero selezionato porta piu' oro nel contesto rispetto al
semplice, tre dove ne porta di meno (caso scomodo da non nascondere: succede in 19 casi su
64, quasi un terzo). Nessuna riusa le domande gia' generate nel pilot di Appunti2 (M2).

Stesso generatore, stesso prompt, due contesti diversi per ciascuna domanda: 12 chiamate.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-generazione/05_confronto_semplice_selezionato.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
M1_OUT = REPO_ROOT / "esperimenti" / "studio-embedding-retrieval" / "output"
OUT_DIR = Path(__file__).resolve().parent / "output"
ENV_PATH = Path(__file__).resolve().parent / "apikey.env"

RERANK_DEPTH = 30
K = 10
MODEL = "gemini-3.1-pro-preview"
BGE = dict(name="BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
           query_prefix="Represent this sentence for searching relevant passages: ")

# scelte fatte guardando output/domande_dove_differisce.json: tre dove il selezionato porta
# piu' oro nel contesto, tre dove ne porta di meno. Diverse conversazioni, nessuna riusata dal pilot.
VINCE = ["4751cd8210b4adb8bce5cbc3fe913096<::>3", "4a8369b7403e54df94d7fa495a4eff27<::>1",
         "4a8369b7403e54df94d7fa495a4eff27<::>2"]
PERDE = ["258193fc88e0e73322e288aa03719260<::>4", "d4d8edb3e0c456d11033b14b28e99470<::>2",
         "7cb06b3f8068674fce61a11e9777f708<::>6"]

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


def generate(client, prompt):
    for tentativo in range(3):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt).text
        except Exception as e:
            if tentativo == 2:
                return f"[ERRORE dopo 3 tentativi: {e}]"
            time.sleep(5)


def main() -> None:
    load_dotenv(ENV_PATH)
    client = genai.Client(api_key=os.environ["GEMINI_API"])

    ids, titles, texts = load_corpus(CORPUS_PATH)
    id_to_idx = {c: i for i, c in enumerate(ids)}
    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    qi = {q: i for i, q in enumerate(query_ids)}
    rewrite = load_variant(MTRAG / "fiqa_rewrite.jsonl")

    d = np.load(M1_OUT / "configurazione_finale_candidati.npz", allow_pickle=True)
    cand = d["cand"]
    sc = np.load(M1_OUT / "configurazione_finale_scores.npy")

    emb_bge = np.load(M1_OUT / "corpus_embeddings_bge_full.npy")
    model_bge = SentenceTransformer(BGE["name"], revision=BGE["revision"], device="cpu",
                                     cache_folder=str(CACHE_HF))

    selezione = [(q, "vince") for q in VINCE] + [(q, "perde") for q in PERDE]
    risultati = []
    for q, esito in selezione:
        i = qi[q]
        gi = {id_to_idx[g] for g in gold_by_query[q] if g in id_to_idx}
        domanda = rewrite[q]

        testa = list(np.argsort(-sc[i][:RERANK_DEPTH], kind="stable"))
        selezionato = [int(cand[i][p]) for p in testa[:K]]

        qv = model_bge.encode([BGE["query_prefix"] + domanda], normalize_embeddings=True,
                               show_progress_bar=False)[0]
        semplice = [int(j) for j in np.argsort(-(emb_bge @ qv), kind="stable")[:K]]

        risposte = {}
        for nome, passaggi in [("selezionato", selezionato), ("semplice", semplice)]:
            blocco = "\n\n".join(f"[{n}] {texts[p]}" for n, p in enumerate(passaggi, start=1))
            prompt = PROMPT_TEMPLATE.format(passages=blocco, question=domanda)
            risposte[nome] = {
                "passaggi": [ids[p] for p in passaggi],
                "oro_nel_contesto": len(gi & set(passaggi)),
                "risposta": generate(client, prompt),
            }

        risultati.append({"query_id": q, "esito_atteso": esito, "domanda": domanda,
                           "n_fonti_annotate": len(gi), **risposte})
        print(f"{q} ({esito}): selezionato oro={risposte['selezionato']['oro_nel_contesto']} "
              f"semplice oro={risposte['semplice']['oro_nel_contesto']}")

    (OUT_DIR / "confronto_semplice_selezionato.json").write_text(
        json.dumps(risultati, indent=2, ensure_ascii=False), encoding="utf-8")

    righe = ["# Confronto semplice / selezionato — 6 domande, stesso generatore", "",
             f"Modello: `{MODEL}`. Tre domande dove il selezionato porta piu' oro nel contesto,",
             "tre dove ne porta di meno. n=6, esplorativo per costruzione.", ""]
    for r in risultati:
        righe += [
            "---", "",
            f"## {r['query_id']} — atteso: il selezionato {r['esito_atteso']}",
            "", f"**Domanda**: {r['domanda']}", "",
            f"**Fonti annotate**: {r['n_fonti_annotate']}", "",
            f"**Selezionato** (oro nel contesto: {r['selezionato']['oro_nel_contesto']})", "",
            r["selezionato"]["risposta"], "",
            f"**Semplice** (oro nel contesto: {r['semplice']['oro_nel_contesto']})", "",
            r["semplice"]["risposta"], "",
        ]
    (OUT_DIR / "confronto_semplice_selezionato.md").write_text("\n".join(righe), encoding="utf-8")
    print("scritto: output/confronto_semplice_selezionato.json, .md")


if __name__ == "__main__":
    main()
