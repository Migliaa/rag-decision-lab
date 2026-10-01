"""Passo 1 di M2: leggere a mano il contesto, prima di generare qualunque cosa.

Il contesto di sessione impone questo come primo passo: prendere poche domande, guardare
il contesto che il recupero selezionato produce davvero, e stabilire *prima* di scrivere
la pipeline che aspetto ha una risposta corretta e come si riconoscono i tre modi di
fallire (la fonte non e' stata trovata / la fonte c'e' ma il contesto non basta / il
contesto basta e la risposta e' sbagliata). In M1 il passo equivalente fu saltato ed e'
costato tre note da correggere.

Qui non si calcola niente di nuovo: si riusano le cache di M1, quindi l'esecuzione dura
secondi e non occupa la macchina.

  - configurazione_finale_candidati.npz : i 60 candidati per domanda della fusione
                                          multi-formulazione, piu' le posizioni dell'oro
  - configurazione_finale_scores.npy    : i punteggi del cross-encoder su quei candidati

Le tre domande non sono scelte a caso: una per ciascun regime di recupero, perche' i tre
modi di fallire si distinguono solo confrontando casi in cui il recupero si comporta in
modo diverso.

  A) oro al primo posto dopo il riordino -> il contesto e' il migliore possibile;
  B) oro presente fra i candidati ma fuori dai primi dieci -> il riordino lo ha perso;
  C) nessun oro fra i 60 candidati -> il recupero ha fallito a monte.

Produce un file leggibile in output/lettura_manuale.md, da annotare a mano.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-generazione/01_lettura_manuale.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
M1_OUT = REPO_ROOT / "esperimenti" / "studio-embedding-retrieval" / "output"
OUT_DIR = Path(__file__).resolve().parent / "output"

RERANK_DEPTH = 30          # profondita' migliore misurata in M1
K = 10                     # quanti passaggi finiscono nel contesto del generatore
MAX_CHARS = 1400           # troncamento in lettura, non nella pipeline


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


def taglia(t: str) -> str:
    t = " ".join(t.split())
    return t if len(t) <= MAX_CHARS else t[:MAX_CHARS].rsplit(" ", 1)[0] + " [...]"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ids, titles, texts = load_corpus(CORPUS_PATH)
    id_to_idx = {c: i for i, c in enumerate(ids)}
    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    rewrite = load_variant(MTRAG / "fiqa_rewrite.jsonl")
    lastturn = load_variant(MTRAG / "fiqa_lastturn.jsonl")

    d = np.load(M1_OUT / "configurazione_finale_candidati.npz", allow_pickle=True)
    cand = d["cand"]
    gold_ranks = [list(x) for x in d["gold_ranks"]]     # posizione nella lista fusa, 1-based
    sc = np.load(M1_OUT / "configurazione_finale_scores.npy")

    # ordine finale dopo il riordino: i primi RERANK_DEPTH riordinati, il resto in coda
    finale = []
    for i in range(len(query_ids)):
        testa = list(np.argsort(-sc[i][:RERANK_DEPTH], kind="stable"))
        coda = list(range(RERANK_DEPTH, cand.shape[1]))
        finale.append([int(cand[i][p]) for p in testa + coda])

    # classificazione di ogni domanda in uno dei tre regimi
    regimi = {"A": [], "B": [], "C": []}
    for i, q in enumerate(query_ids):
        gi = {id_to_idx[g] for g in gold_by_query[q] if g in id_to_idx}
        pos = [finale[i].index(g) + 1 for g in gi if g in finale[i]]
        if pos and min(pos) == 1:
            regimi["A"].append(i)
        elif pos and min(pos) > K:
            regimi["B"].append(i)
        elif not pos:
            regimi["C"].append(i)

    print("domande per regime: " + ", ".join(f"{k}={len(v)}" for k, v in regimi.items()))
    scelte = {k: v[0] for k, v in regimi.items() if v}

    etichette = {
        "A": "oro al primo posto — contesto nella condizione migliore",
        "B": "oro fra i candidati ma fuori dai primi dieci — il riordino lo ha perso",
        "C": "nessun oro fra i 60 candidati — il recupero ha fallito a monte",
    }

    righe = [
        "# Lettura manuale del contesto — tre domande, tre regimi di recupero",
        "",
        "Generato da `01_lettura_manuale.py`. Configurazione di recupero: fusione RRF",
        f"multi-formulazione, riordino con cross-encoder sui primi {RERANK_DEPTH}, primi {K} passaggi",
        "nel contesto. Le annotazioni a mano vanno scritte sotto ogni domanda, nei punti segnati",
        "con `>`.",
        "",
        "Conteggio dei regimi su tutte le 180 domande: "
        + ", ".join(f"**{k}** {len(v)} domande" for k, v in regimi.items())
        + f", restanti {180 - sum(len(v) for v in regimi.values())} con oro fra i primi dieci ma non al primo posto.",
        "",
    ]

    for key, i in scelte.items():
        q = query_ids[i]
        gi = {id_to_idx[g] for g in gold_by_query[q] if g in id_to_idx}
        pos = {g: (finale[i].index(g) + 1 if g in finale[i] else None) for g in gi}
        righe += [
            "---",
            "",
            f"## Regime {key} — {etichette[key]}",
            "",
            f"**Domanda (id `{q}`)**",
            "",
            f"- ultimo turno, cioe' cio' che l'utente scrive davvero: *{lastturn.get(q, '—')}*",
            f"- riscritta da annotatore umano, cioe' cio' che il recupero riceve qui: *{rewrite[q]}*",
            "",
            f"**Fonti annotate come corrette**: {len(gi)}. Posizione finale di ciascuna: "
            + ", ".join(str(p) if p else "fuori dai 60 candidati" for p in pos.values())
            + ".",
            "",
            "> *Che aspetto avrebbe una risposta corretta a questa domanda:*",
            "",
            "> *Come si riconoscerebbe qui un fallimento di generazione (contesto sufficiente, risposta sbagliata):*",
            "",
            "### Le fonti annotate",
            "",
        ]
        for g in sorted(gi):
            righe += [
                f"**[oro] {ids[g]}** — posizione finale: {pos[g] or 'fuori dai candidati'}"
                + (f" — titolo: {titles[g]}" if titles[g] else ""),
                "",
                taglia(texts[g]),
                "",
            ]
        righe += ["### Il contesto che il generatore riceverebbe", ""]
        for rank, g in enumerate(finale[i][:K], start=1):
            marchio = " **(annotata corretta)**" if g in gi else ""
            righe += [
                f"**{rank}. {ids[g]}**{marchio}" + (f" — titolo: {titles[g]}" if titles[g] else ""),
                "",
                taglia(texts[g]),
                "",
            ]
        righe += [
            "> *Dei dieci passaggi qui sopra, quanti rispondono davvero alla domanda (oltre a quelli "
            "annotati)? Le annotazioni di FiQA sono incomplete, e questo conteggio a mano e' l'unico "
            "modo di sapere quanto lo sono su questa domanda:*",
            "",
            "> *Il contesto basta per rispondere?*",
            "",
        ]

    out = OUT_DIR / "lettura_manuale.md"
    out.write_text("\n".join(righe), encoding="utf-8")
    print(f"scritto: {out}")

    riassunto = {
        "profondita_riordino": RERANK_DEPTH,
        "passaggi_nel_contesto": K,
        "conteggio_regimi": {k: len(v) for k, v in regimi.items()},
        "domande_scelte": {k: query_ids[i] for k, i in scelte.items()},
    }
    (OUT_DIR / "lettura_manuale.json").write_text(json.dumps(riassunto, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
