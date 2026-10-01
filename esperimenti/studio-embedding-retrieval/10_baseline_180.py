"""Step 10: rifare la misura di base su tutte e 180 le domande, non sulle 12 del pilota.

Motivo, scoperto in fase di audit: sulle 12 domande del pilota BGE-small risultava
peggiore di MiniLM (0.167 contro 0.264 di recall@10) e tutti i metodi sembravano
collassare sul corpus completo. Su 180 domande l'ordine si inverte e i valori
raddoppiano. Le 12 domande non erano un campione rappresentativo, e ogni conclusione
di design costruita su di esse (Appunti3, Appunti6, Appunti7) va ricalcolata qui.

Aggiunge anche l'intervallo di confidenza bootstrap, che rende visibile quanta
incertezza c'era davvero nei numeri del pilota.

Nessun nuovo embedding del corpus: riusa le matrici cache-ate.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/10_baseline_180.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
REWRITE_PATH = MTRAG / "fiqa_rewrite.jsonl"
PILOT_QRELS_PATH = REPO_ROOT / "data" / "processed" / "E001-pilot-512" / "qrels.tsv"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

RRF_C = 60
K_VALUES = [1, 3, 5, 10, 20, 50, 100, 500]
BOOTSTRAP_SAMPLES = 2000
RNG = np.random.default_rng(0)

WEIGHT_SCHEMES = {
    "uniforme (33/33/33)": {"bm25": 1 / 3, "bge": 1 / 3, "minilm": 1 / 3},
    "denso sopra lessicale (10/45/45)": {"bm25": 0.10, "bge": 0.45, "minilm": 0.45},
    "bge primo (20/50/30)": {"bm25": 0.20, "bge": 0.50, "minilm": 0.30},
}

MODELS = {
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2",
                   revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                   query_prefix="", embeddings_file="corpus_embeddings_minilm_full.npy"),
    "bge": dict(name="BAAI/bge-small-en-v1.5",
                revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                query_prefix="Represent this sentence for searching relevant passages: ",
                embeddings_file="corpus_embeddings_bge_full.npy"),
}


def indexed_text(passage: dict) -> str:
    title = passage.get("title", "")
    text = passage["text"]
    return f"{title}\n{text}" if title else text


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower())


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["_id"])
            texts.append(indexed_text(row))
    return ids, texts


def load_qrels(path: Path) -> dict[str, set[str]]:
    gold: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            query_id, corpus_id, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                gold.setdefault(query_id, set()).add(corpus_id)
    return gold


def load_query_texts(query_ids: set[str], path: Path) -> dict[str, str]:
    texts = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in query_ids:
                texts[row["_id"]] = row["text"].replace("|user|:", "").strip()
    return texts


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def rrf_fuse(rank_arrays: dict[str, np.ndarray], weights: dict[str, float], c: int = RRF_C) -> np.ndarray:
    n = len(next(iter(rank_arrays.values())))
    fused = np.zeros(n, dtype=np.float64)
    for method, ranks in rank_arrays.items():
        fused += weights[method] * (1.0 / (c + ranks))
    return fused


def recall_at_k(gold_ranks: list[int], k: int) -> float:
    return sum(1 for r in gold_ranks if r <= k) / len(gold_ranks)


def ndcg_at_k(gold_ranks: list[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r in gold_ranks if r <= k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_ranks), k) + 1))
    return dcg / idcg if idcg else 0.0


def bootstrap_ci(per_query: list[float], samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    """Intervallo di confidenza al 95% sulla media, ricampionando le domande."""
    arr = np.asarray(per_query)
    idx = RNG.integers(0, len(arr), size=(samples, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    print("Carico il corpus FiQA completo...")
    ids, texts = load_corpus(CORPUS_PATH)
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    print(f"Passaggi: {len(ids)}")

    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    query_texts = load_query_texts(set(query_ids), REWRITE_PATH)
    pilot_query_ids = set(load_qrels(PILOT_QRELS_PATH))
    print(f"Domande valutate: {len(query_ids)} (di cui {len(pilot_query_ids)} erano il pilota)")

    print("Indice BM25...")
    bm25 = BM25Okapi([lexical_tokens(t) for t in texts])

    print("Carico gli embedding gia' calcolati...")
    corpus_embeddings = {k: np.load(OUT_DIR / cfg["embeddings_file"]) for k, cfg in MODELS.items()}
    encoders = {
        k: SentenceTransformer(cfg["name"], revision=cfg["revision"], device="cpu", cache_folder=str(CACHE_HF))
        for k, cfg in MODELS.items()
    }
    query_vectors = {
        k: encoders[k].encode([MODELS[k]["query_prefix"] + query_texts[q] for q in query_ids],
                              normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        for k in MODELS
    }

    methods = ["bm25", "minilm", "bge"] + list(WEIGHT_SCHEMES)
    per_query: dict[str, dict[str, list[float]]] = {
        m: {f"recall@{k}": [] for k in K_VALUES} | {"ndcg@10": []} for m in methods
    }
    gold_ranks_log: dict[str, list[list[int]]] = {m: [] for m in methods}
    evaluated_query_ids: list[str] = []

    for i, query_id in enumerate(query_ids):
        gold_idx = [id_to_idx[g] for g in gold_by_query[query_id] if g in id_to_idx]
        if not gold_idx:
            continue
        evaluated_query_ids.append(query_id)
        text = query_texts[query_id]

        rank_arrays = {"bm25": ranks_from_scores(np.asarray(bm25.get_scores(lexical_tokens(text))))}
        for key in MODELS:
            rank_arrays[key] = ranks_from_scores(corpus_embeddings[key] @ query_vectors[key][i])

        all_ranks = dict(rank_arrays)
        for scheme_name, weights in WEIGHT_SCHEMES.items():
            all_ranks[scheme_name] = ranks_from_scores(rrf_fuse(rank_arrays, weights))

        for method, ranks in all_ranks.items():
            gr = [int(ranks[j]) for j in gold_idx]
            gold_ranks_log[method].append(gr)
            for k in K_VALUES:
                per_query[method][f"recall@{k}"].append(recall_at_k(gr, k))
            per_query[method]["ndcg@10"].append(ndcg_at_k(gr, 10))

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(query_ids)} domande")

    summary = {}
    for method in methods:
        summary[method] = {}
        for metric, vals in per_query[method].items():
            lo, hi = bootstrap_ci(vals)
            summary[method][metric] = {"media": float(np.mean(vals)), "ic95": [lo, hi]}

    print("\n=== 180 domande, corpus completo (intervallo di confidenza 95% bootstrap) ===")
    for method in methods:
        s = summary[method]
        print(f"  {method:<34} recall@10={s['recall@10']['media']:.3f} "
              f"[{s['recall@10']['ic95'][0]:.3f}-{s['recall@10']['ic95'][1]:.3f}]  "
              f"nDCG@10={s['ndcg@10']['media']:.3f}  recall@100={s['recall@100']['media']:.3f}")

    # Confronto diretto: stesse misure ristrette alle 12 domande del pilota.
    pilot_mask = [q in pilot_query_ids for q in evaluated_query_ids]
    print("\n=== Le stesse misure sulle sole 12 domande del pilota, per capire quanto sviava ===")
    pilot_summary = {}
    for method in methods:
        vals = [v for v, keep in zip(per_query[method]["recall@10"], pilot_mask) if keep]
        pilot_summary[method] = float(np.mean(vals))
        print(f"  {method:<34} recall@10={np.mean(vals):.3f}   (su 180: {summary[method]['recall@10']['media']:.3f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "baseline_180.json").write_text(
        json.dumps({
            "n_query": len(query_ids),
            "k_values": K_VALUES,
            "weight_schemes": WEIGHT_SCHEMES,
            "summary_180": summary,
            "summary_pilot12_recall@10": pilot_summary,
            "per_query_recall@10": {m: per_query[m]["recall@10"] for m in methods},
            "query_ids": evaluated_query_ids,
        }, indent=2),
        encoding="utf-8",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    order = ["bm25", "minilm", "bge", "uniforme (33/33/33)", "denso sopra lessicale (10/45/45)", "bge primo (20/50/30)"]
    labels = ["BM25", "MiniLM", "BGE-small", "RRF uniforme", "RRF 10/45/45", "RRF 20/50/30"]
    means = [summary[m]["recall@10"]["media"] for m in order]
    errs = np.array([[summary[m]["recall@10"]["media"] - summary[m]["recall@10"]["ic95"][0] for m in order],
                     [summary[m]["recall@10"]["ic95"][1] - summary[m]["recall@10"]["media"] for m in order]])
    pilot_means = [pilot_summary[m] for m in order]
    x = np.arange(len(order))
    ax1.bar(x - 0.2, pilot_means, width=0.38, color="lightcoral", label="12 domande del pilota")
    ax1.bar(x + 0.2, means, width=0.38, yerr=errs, capsize=4, color="steelblue", label="180 domande (con IC 95%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Recall@10 medio")
    ax1.set_title("Quanto sviava il campione da 12 domande")
    ax1.legend(fontsize=8)

    for method, label in zip(order, labels):
        ax2.plot(K_VALUES, [summary[method][f"recall@{k}"]["media"] for k in K_VALUES], marker="o", label=label)
    ax2.set_xscale("log")
    ax2.set_xticks(K_VALUES)
    ax2.set_xticklabels([str(k) for k in K_VALUES])
    ax2.set_xlabel("k (quanti risultati si guardano)")
    ax2.set_ylabel("Recall@k medio su 180 domande")
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title("Curve di recall, misura di riferimento corretta")

    fig.tight_layout()
    out_path = OUT_DIR / "baseline_180.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out_path}")


if __name__ == "__main__":
    main()
