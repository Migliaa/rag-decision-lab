"""Step 12b: confronto completo delle strategie di reranking, con piu' reranker.

Copiato da 12_strategie_reranking.py ed esteso dopo due risultati dello step 12:
- rerankare 50 candidati batte rerankarne 100 o 200, quindi la profondita' va
  esplorata fitta invece che a tre valori;
- con un solo reranker non si potevano misurare ne' la fusione fra reranker ne' la
  cascata, che sono due delle domande poste.

Tutte le strategie leggono punteggi gia' calcolati: il confronto costa secondi.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/12b_strategie_reranking_complete.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "output"
RRF_C = 60
K = 10
DEPTHS = [10, 20, 30, 50, 75, 100]
RNG = np.random.default_rng(0)
BGE_EMB = OUT_DIR / "corpus_embeddings_bge_full.npy"


def load_inputs():
    data = np.load(OUT_DIR / "candidati_180.npz", allow_pickle=True)
    candidates = data["candidates"]
    gold_ranks_full = [list(g) for g in data["gold_ranks_full"]]
    query_ids = list(data["query_ids"])
    scores = {}
    for path in sorted(OUT_DIR.glob("rerank_scores_*.npy")):
        key = path.stem.replace("rerank_scores_", "")
        prog = json.loads((OUT_DIR / f"rerank_progress_{key}.json").read_text())
        if prog["done"] < len(query_ids):
            print(f"  {key}: incompleto ({prog['done']}/{len(query_ids)}), escluso.")
            continue
        scores[key] = np.load(path)
    return candidates, gold_ranks_full, query_ids, scores


def minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def ranks_of(scores):
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def metrics(new_order, gold_ranks_full, depth):
    pos = {int(p): i + 1 for i, p in enumerate(new_order)}
    final = [pos[r - 1] if r <= depth else r for r in gold_ranks_full]
    recall = sum(1 for r in final if r <= K) / len(final)
    dcg = sum(1.0 / math.log2(r + 1) for r in final if r <= K)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(final), K) + 1))
    return recall, (dcg / idcg if idcg else 0.0)


def evaluate(fn, golds, depth, subset=None):
    rec, nd = [], []
    for i, g in enumerate(golds):
        if subset is not None and i not in subset:
            continue
        r, n = metrics(fn(i), g, depth)
        rec.append(r)
        nd.append(n)
    return float(np.mean(rec)), float(np.mean(nd)), rec


def ci(vals, samples=2000):
    a = np.asarray(vals)
    idx = RNG.integers(0, len(a), size=(samples, len(a)))
    m = a[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    candidates, golds, query_ids, scores = load_inputs()
    n_q = len(query_ids)
    keys = list(scores)
    depths_avail = {k: v.shape[1] for k, v in scores.items()}
    print(f"Domande: {n_q}   reranker: {keys}   profondita' disponibili: {depths_avail}")

    results = {}

    def record(name, r, n, pq, depth=None):
        lo, hi = ci(pq)
        results[name] = {"recall@10": r, "ndcg@10": n, "ic95": [lo, hi], "depth": depth}
        print(f"  {name:<56} recall@10={r:.3f} [{lo:.3f}-{hi:.3f}]  nDCG@10={n:.3f}")

    print("\n=== 0. Riferimento senza reranking, e tetto della lista ===")
    r, n, pq = evaluate(lambda i: np.arange(10), golds, 10)
    record("nessun reranking (ordine della fusione)", r, n, pq)
    for d in DEPTHS:
        ceiling = np.mean([sum(1 for x in g if x <= d) / len(g) for g in golds])
        print(f"    tetto con {d:3d} candidati: {ceiling:.3f}")

    print("\n=== 1. Riordino puro: reranker x profondita' ===")
    best = (None, -1)
    for key in keys:
        for d in [x for x in DEPTHS if x <= depths_avail[key]]:
            r, n, pq = evaluate(lambda i, s=scores[key], d=d: np.argsort(-s[i][:d], kind="stable"), golds, d)
            record(f"{key} sui primi {d}", r, n, pq, d)
            if n > best[1]:
                best = ((key, d), n)
    (best_key, best_depth) = best[0]
    print(f"  migliore singolo: {best_key} a profondita' {best_depth}")

    print("\n=== 2. Fondere piu' reranker (RRF sugli ordinamenti) ===")
    if len(keys) >= 2:
        common = min(depths_avail.values())
        for d in [x for x in DEPTHS if x <= common]:
            def fuse(i, d=d, ks=keys):
                tot = np.zeros(d)
                for k in ks:
                    tot += 1.0 / (RRF_C + ranks_of(scores[k][i][:d]))
                return np.argsort(-tot, kind="stable")
            r, n, pq = evaluate(fuse, golds, d)
            record(f"RRF fra {len(keys)} reranker, primi {d}", r, n, pq, d)

        # anche a coppie, per vedere se la coppia di famiglie diverse batte la coppia simile
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ka, kb = keys[a], keys[b]
                d = min(depths_avail[ka], depths_avail[kb], best_depth)
                def fuse2(i, ka=ka, kb=kb, d=d):
                    tot = 1.0 / (RRF_C + ranks_of(scores[ka][i][:d])) + 1.0 / (RRF_C + ranks_of(scores[kb][i][:d]))
                    return np.argsort(-tot, kind="stable")
                r, n, pq = evaluate(fuse2, golds, d)
                record(f"RRF {ka} + {kb}, primi {d}", r, n, pq, d)

    print("\n=== 3. Cascata: filtro economico su lista larga, secondo modello sui sopravvissuti ===")
    cheap = "ms-marco-MiniLM-L6"
    if cheap in scores:
        for exp in [k for k in keys if k != cheap]:
            d_exp = depths_avail[exp]
            for wide in (100, 200):
                if wide > depths_avail[cheap]:
                    continue
                for surv in (20, 30, 50):
                    def cascade(i, e=exp, w=wide, s=surv, de=d_exp):
                        order = np.argsort(-scores[cheap][i][:w], kind="stable")
                        eligible = [int(p) for p in order if p < de]
                        keep = eligible[:s]
                        keepset = set(keep)
                        rest = [int(p) for p in order if p not in keepset]
                        reordered = sorted(keep, key=lambda p: -scores[e][i][p])
                        return np.array(reordered + rest)
                    r, n, pq = evaluate(cascade, golds, wide)
                    record(f"cascata {cheap}({wide}) -> {exp}(primi {surv})", r, n, pq, wide)

    print("\n=== 4. Interpolazione con il rango della prima fase, alla profondita' migliore ===")
    dev, test = set(range(0, n_q, 2)), set(range(1, n_q, 2))
    d = best_depth
    first_stage = 1.0 / (RRF_C + np.arange(1, d + 1))
    alphas = np.round(np.arange(0.0, 1.01, 0.1), 2)
    dev_res = []
    for a in alphas:
        def strat(i, a=a, d=d, s=scores[best_key]):
            return np.argsort(-(a * minmax(s[i][:d]) + (1 - a) * minmax(first_stage)), kind="stable")
        _, n, _ = evaluate(strat, golds, d, subset=dev)
        dev_res.append((n, a))
    dev_res.sort(reverse=True)
    a_star = dev_res[0][1]
    print(f"  alpha scelto sulle domande pari: {a_star}")
    def strat_star(i, a=a_star, d=d, s=scores[best_key]):
        return np.argsort(-(a * minmax(s[i][:d]) + (1 - a) * minmax(first_stage)), kind="stable")
    r, n, pq = evaluate(strat_star, golds, d, subset=test)
    record(f"interpolato alpha={a_star} su {best_key}({d}) - verifica su domande dispari", r, n, pq, d)

    print("\n=== 5. Diversificazione MMR alla profondita' migliore ===")
    if BGE_EMB.exists():
        emb = np.load(BGE_EMB, mmap_mode="r")
        for lam in (0.7, 0.9, 0.95):
            def mmr(i, d=best_depth, lam=lam, s=scores[best_key]):
                vecs = np.asarray(emb[candidates[i][:d]], dtype=np.float32)
                sim = vecs @ vecs.T
                rel = minmax(s[i][:d])
                max_sim = np.zeros(d, dtype=np.float32)
                avail = np.ones(d, dtype=bool)
                chosen = []
                for step in range(d):
                    sc = rel if step == 0 else lam * rel - (1 - lam) * max_sim
                    pick = int(np.argmax(np.where(avail, sc, -np.inf)))
                    chosen.append(pick)
                    avail[pick] = False
                    max_sim = np.maximum(max_sim, sim[pick])
                return np.array(chosen)
            r, n, pq = evaluate(mmr, golds, best_depth)
            record(f"MMR lambda={lam} su {best_key}({best_depth})", r, n, pq, best_depth)

    (OUT_DIR / "strategie_reranking_complete.json").write_text(
        json.dumps({"n_query": n_q, "reranker": keys, "migliore_singolo": [best_key, best_depth],
                    "alpha": float(a_star), "risultati": results}, indent=2), encoding="utf-8")

    ordered = sorted(results.items(), key=lambda kv: kv[1]["ndcg@10"])
    names = [k for k, _ in ordered]
    nd = [v["ndcg@10"] for _, v in ordered]
    rc = [v["recall@10"] for _, v in ordered]
    fig, ax = plt.subplots(figsize=(12, 0.34 * len(names) + 2.4))
    y = np.arange(len(names))
    ax.barh(y - 0.2, nd, height=0.38, color="steelblue", label="nDCG@10")
    ax.barh(y + 0.2, rc, height=0.38, color="lightsteelblue", label="Recall@10")
    base = results["nessun reranking (ordine della fusione)"]
    ax.axvline(base["ndcg@10"], color="crimson", linestyle="--", linewidth=1, label="nDCG@10 senza reranking")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("valore medio su 180 domande")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Strategie di reranking, stessi candidati, 180 domande")
    fig.tight_layout()
    out = OUT_DIR / "strategie_reranking_complete.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out}")


if __name__ == "__main__":
    main()
