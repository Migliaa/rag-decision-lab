"""Step 12: confrontare strategie di reranking diverse sugli stessi candidati.

Tutti i punteggi costosi sono gia' su disco (step 11): qui si combinano soltanto,
quindi ogni strategia costa millisecondi e si possono confrontare tutte insieme.

Le strategie rispondono a domande di design precise:
- profondita': rerankare 50, 100 o 200 candidati cambia il risultato?
- riordino puro contro interpolazione con il punteggio di prima fase: fidarsi
  ciecamente del cross-encoder o tenere memoria di quanto diceva il recupero?
- fondere piu' reranker tra loro (RRF sui loro ordinamenti) invece di sceglierne uno
- cascata: reranker piccolo e veloce su lista larga, poi reranker grande solo sui
  sopravvissuti - il "giro doppio" a costo controllato
- diversificazione (MMR): penalizzare i candidati troppo simili tra loro per non
  riempire le prime posizioni con dieci varianti dello stesso contenuto

Le strategie con un parametro libero (alpha, lambda) vengono scelte su meta' delle
domande e misurate sull'altra meta', per non tarare e dichiarare vittoria sugli
stessi dati.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/12_strategie_reranking.py
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
K_REPORT = 10
RNG = np.random.default_rng(0)

BGE_EMB = OUT_DIR / "corpus_embeddings_bge_full.npy"


def load_inputs():
    data = np.load(OUT_DIR / "candidati_180.npz", allow_pickle=True)
    candidates = data["candidates"]
    gold_ranks_full = [list(g) for g in data["gold_ranks_full"]]
    query_ids = list(data["query_ids"])

    rerank_scores = {}
    for path in sorted(OUT_DIR.glob("rerank_scores_*.npy")):
        key = path.stem.replace("rerank_scores_", "")
        progress = json.loads((OUT_DIR / f"rerank_progress_{key}.json").read_text())
        if progress["done"] < len(query_ids):
            print(f"  ATTENZIONE: {key} incompleto ({progress['done']}/{len(query_ids)}), lo salto.")
            continue
        rerank_scores[key] = np.load(path)
    return candidates, gold_ranks_full, query_ids, rerank_scores


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def ranks_of(scores: np.ndarray) -> np.ndarray:
    """Rank 1-indicizzato (1 = punteggio piu' alto) per ogni posizione dell'array."""
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def metrics_from_order(new_order: np.ndarray, gold_ranks_full: list[int], depth: int, k: int = K_REPORT):
    """new_order: posizioni originali (0-indicizzate nella lista candidati) nel nuovo ordine.

    L'oro fuori dai primi `depth` candidati conserva il rank che aveva nella fusione.
    """
    position_of = {int(p): i + 1 for i, p in enumerate(new_order)}
    final = []
    for r in gold_ranks_full:
        if r <= depth:
            final.append(position_of[r - 1])
        else:
            final.append(r)
    recall = sum(1 for r in final if r <= k) / len(final)
    dcg = sum(1.0 / math.log2(r + 1) for r in final if r <= k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(final), k) + 1))
    return recall, (dcg / idcg if idcg else 0.0)


def evaluate(strategy_fn, gold_ranks_all, depth, query_subset=None):
    recalls, ndcgs = [], []
    for i, gold in enumerate(gold_ranks_all):
        if query_subset is not None and i not in query_subset:
            continue
        order = strategy_fn(i)
        r, n = metrics_from_order(order, gold, depth)
        recalls.append(r)
        ndcgs.append(n)
    return float(np.mean(recalls)), float(np.mean(ndcgs)), recalls


def bootstrap_ci(vals, samples=2000):
    arr = np.asarray(vals)
    idx = RNG.integers(0, len(arr), size=(samples, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    candidates, gold_ranks_full, query_ids, rerank_scores = load_inputs()
    n_q = len(query_ids)
    print(f"Domande: {n_q}   reranker disponibili: {list(rerank_scores)}")

    depths = {k: v.shape[1] for k, v in rerank_scores.items()}
    print(f"Profondita' dei punteggi: {depths}")

    results = {}

    def record(name, recall, ndcg, per_query, depth):
        lo, hi = bootstrap_ci(per_query)
        results[name] = {"recall@10": recall, "ndcg@10": ndcg, "ic95": [lo, hi], "depth": depth}
        print(f"  {name:<52} recall@10={recall:.3f} [{lo:.3f}-{hi:.3f}]  nDCG@10={ndcg:.3f}")

    # --- riferimento: nessun reranking, ordine della fusione ---
    max_depth = max(depths.values())
    base_recall, base_ndcg, base_per_q = evaluate(
        lambda i: np.arange(max_depth), gold_ranks_full, max_depth)
    print("\n=== Riferimento ===")
    record("fusione RRF senza reranking", base_recall, base_ndcg, base_per_q, max_depth)

    # --- 1. reranking puro, per modello e per profondita' ---
    print("\n=== 1. Riordino puro: quale reranker, quanto in profondita' ===")
    for key, scores in rerank_scores.items():
        for depth in sorted({50, 100, depths[key]}):
            if depth > depths[key]:
                continue
            r, n, pq = evaluate(lambda i, s=scores, d=depth: np.argsort(-s[i][:d], kind="stable"),
                                gold_ranks_full, depth)
            record(f"{key} sui primi {depth}", r, n, pq, depth)

    # --- 2. interpolazione con il punteggio di prima fase ---
    print("\n=== 2. Interpolazione con la prima fase (alpha=0 solo fusione, 1 solo reranker) ===")
    dev = set(range(0, n_q, 2))          # domande pari: taratura
    test = set(range(1, n_q, 2))         # domande dispari: verifica
    fusion_proxy = {}  # punteggio di prima fase ricostruito dal rank: 1/(c+rank)
    for key, scores in rerank_scores.items():
        d = depths[key]
        fusion_proxy[key] = np.tile(1.0 / (RRF_C + np.arange(1, d + 1)), (n_q, 1))

    best_alpha = {}
    for key, scores in rerank_scores.items():
        d = depths[key]
        fus = fusion_proxy[key]
        alphas = np.round(np.arange(0.0, 1.01, 0.1), 2)
        dev_scores = []
        for a in alphas:
            def strat(i, s=scores, f=fus, a=a, d=d):
                return np.argsort(-(a * minmax(s[i][:d]) + (1 - a) * minmax(f[i][:d])), kind="stable")
            r, n, _ = evaluate(strat, gold_ranks_full, d, query_subset=dev)
            dev_scores.append((n, r, a))
        dev_scores.sort(reverse=True)
        a_star = dev_scores[0][2]
        best_alpha[key] = float(a_star)

        def strat_star(i, s=scores, f=fus, a=a_star, d=d):
            return np.argsort(-(a * minmax(s[i][:d]) + (1 - a) * minmax(f[i][:d])), kind="stable")
        r_t, n_t, pq_t = evaluate(strat_star, gold_ranks_full, d, query_subset=test)
        r_all, n_all, pq_all = evaluate(strat_star, gold_ranks_full, d)
        print(f"  alpha scelto su meta' domande per {key}: {a_star}")
        record(f"{key} interpolato (alpha={a_star}) - solo meta' di verifica", r_t, n_t, pq_t, d)
        record(f"{key} interpolato (alpha={a_star}) - tutte le domande", r_all, n_all, pq_all, d)

    # --- 3. fusione di piu' reranker (RRF sui loro ordinamenti) ---
    print("\n=== 3. Fondere piu' reranker invece di sceglierne uno ===")
    keys = list(rerank_scores)
    if len(keys) >= 2:
        common = min(depths.values())

        def fuse_rerankers(i, ks=keys, d=common):
            total = np.zeros(d)
            for k in ks:
                total += 1.0 / (RRF_C + ranks_of(rerank_scores[k][i][:d]))
            return np.argsort(-total, kind="stable")
        r, n, pq = evaluate(fuse_rerankers, gold_ranks_full, common)
        record(f"RRF fra {len(keys)} reranker (primi {common})", r, n, pq, common)

        # fusione che include anche la prima fase come "votante"
        def fuse_with_first_stage(i, ks=keys, d=common):
            total = 1.0 / (RRF_C + np.arange(1, d + 1))
            for k in ks:
                total += 1.0 / (RRF_C + ranks_of(rerank_scores[k][i][:d]))
            return np.argsort(-total, kind="stable")
        r, n, pq = evaluate(fuse_with_first_stage, gold_ranks_full, common)
        record(f"RRF fra {len(keys)} reranker + prima fase (primi {common})", r, n, pq, common)

    # --- 4. cascata: reranker economico su lista larga, reranker grande sui sopravvissuti ---
    print("\n=== 4. Cascata: filtro economico su lista larga, modello grande sui sopravvissuti ===")
    cheap = "ms-marco-MiniLM-L6"
    if cheap in rerank_scores:
        for expensive in [k for k in rerank_scores if k != cheap]:
            d_exp = depths[expensive]
            d_cheap = depths[cheap]
            for survivors in (25, 50):
                def cascade(i, e=expensive, s=survivors, dc=d_cheap, de=d_exp):
                    cheap_order = np.argsort(-rerank_scores[cheap][i][:dc], kind="stable")
                    keep = [p for p in cheap_order if p < de][:s]
                    rest = [p for p in cheap_order if p not in set(keep)]
                    reordered = sorted(keep, key=lambda p: -rerank_scores[e][i][p])
                    return np.array(reordered + rest)
                r, n, pq = evaluate(cascade, gold_ranks_full, d_cheap)
                record(f"cascata: {cheap} su {d_cheap} -> {expensive} sui primi {survivors}", r, n, pq, d_cheap)

    # --- 5. diversificazione MMR sui risultati del miglior reranker ---
    print("\n=== 5. Diversificazione (MMR): penalizzare candidati troppo simili tra loro ===")
    if BGE_EMB.exists() and rerank_scores:
        emb = np.load(BGE_EMB, mmap_mode="r")
        best_key = max(rerank_scores, key=lambda k: results.get(f"{k} sui primi {depths[k]}", {}).get("ndcg@10", 0))
        d = depths[best_key]
        print(f"  applicata sopra: {best_key}")
        for lam in (0.5, 0.7, 0.9):
            def mmr(i, key=best_key, d=d, lam=lam):
                """Selezione golosa: rilevanza meno somiglianza massima con i gia' scelti.

                La somiglianza massima con l'insieme scelto si aggiorna in modo
                incrementale, altrimenti il costo diventa cubico nella lunghezza
                della lista.
                """
                vecs = np.asarray(emb[candidates[i][:d]], dtype=np.float32)
                sim = vecs @ vecs.T
                rel = minmax(rerank_scores[key][i][:d])
                max_sim = np.full(d, -np.inf, dtype=np.float32)
                available = np.ones(d, dtype=bool)
                chosen = []
                for _ in range(d):
                    if not chosen:
                        score = rel.copy()
                    else:
                        score = lam * rel - (1 - lam) * max_sim
                    score = np.where(available, score, -np.inf)
                    pick = int(np.argmax(score))
                    chosen.append(pick)
                    available[pick] = False
                    max_sim = np.maximum(max_sim, sim[pick]) if chosen else sim[pick]
                return np.array(chosen)
            r, n, pq = evaluate(mmr, gold_ranks_full, d)
            record(f"MMR (lambda={lam}) sopra {best_key}", r, n, pq, d)

    (OUT_DIR / "strategie_reranking.json").write_text(
        json.dumps({"n_query": n_q, "depths": depths, "alpha_scelti": best_alpha, "risultati": results}, indent=2),
        encoding="utf-8")

    # --- figura ---
    ordered = sorted(results.items(), key=lambda kv: kv[1]["ndcg@10"])
    names = [k for k, _ in ordered]
    vals = [v["ndcg@10"] for _, v in ordered]
    rec = [v["recall@10"] for _, v in ordered]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(names) + 2.2))
    y = np.arange(len(names))
    ax.barh(y - 0.2, vals, height=0.38, color="steelblue", label="nDCG@10")
    ax.barh(y + 0.2, rec, height=0.38, color="lightsteelblue", label="Recall@10")
    base_idx = [i for i, nm in enumerate(names) if nm == "fusione RRF senza reranking"]
    if base_idx:
        ax.axvline(vals[base_idx[0]], color="crimson", linestyle="--", linewidth=1,
                   label="nDCG@10 senza reranking")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("valore medio su 180 domande")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Strategie di reranking a confronto, stessi candidati")
    fig.tight_layout()
    out_path = OUT_DIR / "strategie_reranking.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out_path}")


if __name__ == "__main__":
    main()
