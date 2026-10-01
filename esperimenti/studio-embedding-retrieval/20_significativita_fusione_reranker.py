"""Step 20: la fusione fra reranker di famiglie diverse e' un guadagno vero o rumore?

Lo step 19 mostra che fondere due cross-encoder di famiglie diverse (MiniLM e DeBERTa-v3)
porta nDCG@10 da 0.389 a 0.417, mentre fonderne due della stessa famiglia lo lascia dov'e'.
Gli intervalli di confidenza dei due valori si sovrappongono ampiamente, ma quelli sono
intervalli sulla media di ciascun metodo e non dicono niente su una differenza misurata
sulle *stesse* domande: due metodi valutati sullo stesso campione sbagliano insieme, e la
differenza appaiata ha una variabilita' molto minore della differenza fra le due medie.

Qui si guarda la differenza domanda per domanda, con due strumenti che non assumono una
distribuzione: un bootstrap appaiato sulla differenza media e il test dei segni per ranghi
di Wilcoxon. Si contano anche le domande migliorate e peggiorate, che e' il modo piu' onesto
di presentare un guadagno medio: una media puo' nascere da un miglioramento diffuso o da
pochi casi estremi, e sono due cose diverse.

Il confronto e' a parita' di candidati e di profondita' (i primi 30 della stessa lista fusa).

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/20_significativita_fusione_reranker.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT_DIR = Path(__file__).resolve().parent / "output"
RRF_C = 60
K = 10
DEPTH = 30
RNG = np.random.default_rng(0)


def ranks_of(s):
    order = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.int64)
    r[order] = np.arange(1, len(s) + 1)
    return r


def metrics(new_order, gold_ranks_full, depth):
    pos = {int(p): i + 1 for i, p in enumerate(new_order)}
    final = [pos[r - 1] if r <= depth else r for r in gold_ranks_full]
    recall = sum(1 for r in final if r <= K) / len(final)
    dcg = sum(1.0 / math.log2(r + 1) for r in final if r <= K)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(final), K) + 1))
    return recall, (dcg / idcg if idcg else 0.0)


def per_query(fn, golds):
    rec, nd = [], []
    for i, g in enumerate(golds):
        r, n = metrics(fn(i), g, DEPTH)
        rec.append(r)
        nd.append(n)
    return np.asarray(rec), np.asarray(nd)


def confronta(nome, a, b, differenze_out):
    """a = metodo nuovo, b = riferimento; entrambi vettori per domanda."""
    d = a - b
    idx = RNG.integers(0, len(d), size=(10000, len(d)))
    medie = d[idx].mean(axis=1)
    lo, hi = float(np.percentile(medie, 2.5)), float(np.percentile(medie, 97.5))
    meglio = int((d > 1e-12).sum())
    peggio = int((d < -1e-12).sum())
    pari = len(d) - meglio - peggio
    if meglio + peggio > 0:
        stat, p = wilcoxon(a, b, zero_method="wilcox")
        p = float(p)
    else:
        p = 1.0
    print(f"\n{nome}")
    print(f"  media nuovo {a.mean():.4f}   media riferimento {b.mean():.4f}   differenza {d.mean():+.4f}")
    print(f"  intervallo 95% sulla differenza appaiata: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  domande migliorate {meglio}, peggiorate {peggio}, invariate {pari}")
    print(f"  Wilcoxon p = {p:.5f}")
    differenze_out[nome] = {
        "media_nuovo": float(a.mean()), "media_riferimento": float(b.mean()),
        "differenza": float(d.mean()), "ic95_differenza": [lo, hi],
        "migliorate": meglio, "peggiorate": peggio, "invariate": pari, "wilcoxon_p": p,
    }


def main() -> None:
    data = np.load(OUT_DIR / "candidati_180.npz", allow_pickle=True)
    golds = [list(g) for g in data["gold_ranks_full"]]
    s = {k: np.load(OUT_DIR / f"rerank_scores_{k}.npy")
         for k in ["ms-marco-MiniLM-L6", "ms-marco-MiniLM-L12", "mxbai-rerank-xsmall"]}

    def solo(k):
        return lambda i: np.argsort(-s[k][i][:DEPTH], kind="stable")

    def fusi(ks):
        def f(i):
            tot = np.zeros(DEPTH)
            for k in ks:
                tot += 1.0 / (RRF_C + ranks_of(s[k][i][:DEPTH]))
            return np.argsort(-tot, kind="stable")
        return f

    rif_r, rif_n = per_query(solo("ms-marco-MiniLM-L6"), golds)
    risultati = {}

    for nome, fn in [
        ("stessa famiglia: RRF L6 + L12 contro L6 da solo",
         fusi(["ms-marco-MiniLM-L6", "ms-marco-MiniLM-L12"])),
        ("famiglie diverse: RRF L6 + mxbai contro L6 da solo",
         fusi(["ms-marco-MiniLM-L6", "mxbai-rerank-xsmall"])),
        ("tutti e tre: RRF L6 + L12 + mxbai contro L6 da solo",
         fusi(["ms-marco-MiniLM-L6", "ms-marco-MiniLM-L12", "mxbai-rerank-xsmall"])),
        ("mxbai da solo contro L6 da solo", solo("mxbai-rerank-xsmall")),
    ]:
        a_r, a_n = per_query(fn, golds)
        confronta(f"[nDCG@10] {nome}", a_n, rif_n, risultati)
        confronta(f"[recall@10] {nome}", a_r, rif_r, risultati)

    out = OUT_DIR / "significativita_fusione_reranker.json"
    out.write_text(json.dumps({"profondita": DEPTH, "confronti": risultati}, indent=2), encoding="utf-8")
    print(f"\nScritto: {out}")


if __name__ == "__main__":
    main()
