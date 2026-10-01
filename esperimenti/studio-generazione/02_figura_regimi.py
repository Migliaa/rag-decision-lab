"""Passo 2 di M2: quante domande hanno davvero una fonte annotata nel contesto.

Il recall@10 di M1 (0.489) conta le *fonti* recuperate sul totale delle fonti annotate.
Per la generazione conta un'altra cosa: quante *domande* ricevono almeno una fonte
annotata nel contesto, perche' e' la domanda, non la fonte, l'unita' su cui si giudica
una risposta. Le due quote non coincidono, e la seconda non era mai stata calcolata.

Riusa le stesse cache di M1 del passo 1, nessun calcolo pesante.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-generazione/02_figura_regimi.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
M1_OUT = REPO_ROOT / "esperimenti" / "studio-embedding-retrieval" / "output"
OUT_DIR = Path(__file__).resolve().parent / "output"

RERANK_DEPTH = 30
K = 10


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ids = [json.loads(l)["_id"] for l in CORPUS_PATH.open(encoding="utf-8")]
    id_to_idx = {c: i for i, c in enumerate(ids)}

    gold_by_query = {}
    with DEV_QRELS.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            q, c, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                gold_by_query.setdefault(q, set()).add(c)
    query_ids = sorted(gold_by_query)

    d = np.load(M1_OUT / "configurazione_finale_candidati.npz", allow_pickle=True)
    cand = d["cand"]
    sc = np.load(M1_OUT / "configurazione_finale_scores.npy")

    conteggi = {"primo posto": 0, "nei primi dieci": 0, "fra i candidati, oltre il decimo": 0,
                "assente dai candidati": 0}
    quote_nel_contesto = []
    for i, q in enumerate(query_ids):
        testa = list(np.argsort(-sc[i][:RERANK_DEPTH], kind="stable"))
        ordine = [int(cand[i][p]) for p in testa + list(range(RERANK_DEPTH, cand.shape[1]))]
        gi = {id_to_idx[g] for g in gold_by_query[q] if g in id_to_idx}
        pos = [ordine.index(g) + 1 for g in gi if g in ordine]
        quote_nel_contesto.append(sum(1 for p in pos if p <= K) / len(gi))
        if pos and min(pos) == 1:
            conteggi["primo posto"] += 1
        elif pos and min(pos) <= K:
            conteggi["nei primi dieci"] += 1
        elif pos:
            conteggi["fra i candidati, oltre il decimo"] += 1
        else:
            conteggi["assente dai candidati"] += 1

    tot = len(query_ids)
    con_contesto = conteggi["primo posto"] + conteggi["nei primi dieci"]
    quota_fonti = float(np.mean(quote_nel_contesto))
    print(f"domande con almeno una fonte annotata nel contesto: {con_contesto}/{tot} = {con_contesto/tot:.3f}")
    print(f"quota media di fonti annotate che entrano nel contesto: {quota_fonti:.3f}")

    etichette = list(conteggi)
    valori = [conteggi[e] for e in etichette]
    colori = ["#2f6f4e", "#6aa87f", "#d8a657", "#b4544a"]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    sinistra = 0
    for e, v, c in zip(etichette, valori, colori):
        ax.barh([0], [v], left=sinistra, color=c, edgecolor="white")
        ax.text(sinistra + v / 2, 0, f"{v}\n{v/tot:.0%}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        sinistra += v
    ax.set_yticks([])
    ax.set_xlim(0, tot)
    ax.set_xlabel(f"domande (su {tot})")
    ax.set_title("Posizione della migliore fonte annotata, configurazione di recupero selezionata")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colori], etichette,
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2, frameon=False)
    fig.tight_layout()
    out = OUT_DIR / "regimi_recupero.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"figura: {out}")

    (OUT_DIR / "regimi_recupero.json").write_text(json.dumps({
        "conteggi": conteggi,
        "domande_con_fonte_nel_contesto": con_contesto / tot,
        "quota_media_fonti_nel_contesto": quota_fonti,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
