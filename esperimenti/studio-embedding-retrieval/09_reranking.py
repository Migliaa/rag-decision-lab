"""Step 9: reranking con cross-encoder sui top-100 della fusione RRF uniforme.

Copiato da 08_fusione_rrf.py e ristretto a un solo schema di pesi (uniforme,
la decisione presa in Appunti6). Aggiunge un passo in più: i 100 candidati
più alti della lista fusa vengono riletti da un cross-encoder (BGE-reranker,
variante large) che guarda query e passaggio insieme, non separatamente come
fanno gli embedding bi-encoder. Il costo è delimitato: 100 candidati per
domanda, 12 domande, non il corpus intero.

Nessun nuovo embedding bi-encoder: riusa le stesse matrici cache-ate del
corpus completo. Il cross-encoder è nuovo e viene scaricato al primo avvio.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/09_reranking.py
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
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
PILOT_QRELS_PATH = REPO_ROOT / "data" / "processed" / "E001-pilot-512" / "qrels.tsv"
REWRITE_PATH = (
    REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa" / "fiqa_rewrite.jsonl"
)
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

RRF_C = 60
RERANK_K = 100  # taglio deciso in Appunti6: 100 candidati, non l'intero corpus
K_VALUES = [1, 3, 5, 10, 20, 50, 100]  # dove misurare l'effetto del rerank, tutto dentro i 100 candidati

UNIFORM_WEIGHTS = {"bm25": 1 / 3, "bge": 1 / 3, "minilm": 1 / 3}

MODELS = {
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2",
                   revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                   query_prefix="", embeddings_file="corpus_embeddings_minilm_full.npy"),
    "bge": dict(name="BAAI/bge-small-en-v1.5",
                revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                query_prefix="Represent this sentence for searching relevant passages: ",
                embeddings_file="corpus_embeddings_bge_full.npy"),
}

# Reranker cross-encoder: variante "large" di BGE, come proposto da Andrea.
RERANKER_NAME = "BAAI/bge-reranker-large"
RERANKER_REVISION = "55611d7bca2a7133960a6d3b71e083071bbfc312"


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


def load_pilot_query_gold() -> dict[str, set[str]]:
    gold_by_query: dict[str, set[str]] = {}
    with PILOT_QRELS_PATH.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            query_id, corpus_id, _score = line.rstrip("\n").split("\t")
            gold_by_query.setdefault(query_id, set()).add(corpus_id)
    return gold_by_query


def load_rewrite_texts(query_ids: set[str], path: Path) -> dict[str, str]:
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


def recall_at_k_from_ranks(gold_ranks: list[int], k: int) -> float:
    return sum(1 for r in gold_ranks if r <= k) / len(gold_ranks)


def ndcg_at_k_from_ranks(gold_ranks: list[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r in gold_ranks if r <= k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_ranks), k) + 1))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    print("Carico il corpus FiQA completo...")
    ids, texts = load_corpus(CORPUS_PATH)
    n = len(ids)
    print(f"Passaggi: {n}")

    gold_by_query = load_pilot_query_gold()
    query_ids = sorted(gold_by_query)
    query_texts = load_rewrite_texts(set(query_ids), REWRITE_PATH)

    print("Indice BM25...")
    bm25 = BM25Okapi([lexical_tokens(t) for t in texts])

    print("Carico gli embedding del corpus già calcolati (nessun nuovo calcolo sul corpus)...")
    corpus_embeddings = {key: np.load(OUT_DIR / cfg["embeddings_file"]) for key, cfg in MODELS.items()}
    encoders = {
        key: SentenceTransformer(cfg["name"], revision=cfg["revision"], device="cpu", cache_folder=str(CACHE_HF))
        for key, cfg in MODELS.items()
    }

    print(f"Carico il reranker {RERANKER_NAME} (primo avvio: scarica ~1.1 GB)...")
    reranker = CrossEncoder(RERANKER_NAME, revision=RERANKER_REVISION, device="cpu", cache_folder=str(CACHE_HF))

    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    per_query_results = []

    for query_id in query_ids:
        gold_ids = gold_by_query[query_id]
        gold_idx = [id_to_idx[g] for g in gold_ids]
        text = query_texts[query_id]

        bm25_scores = np.asarray(bm25.get_scores(lexical_tokens(text)))
        rank_arrays = {"bm25": ranks_from_scores(bm25_scores)}
        for key, cfg in MODELS.items():
            qvec = encoders[key].encode([cfg["query_prefix"] + text], normalize_embeddings=True)[0]
            scores = corpus_embeddings[key] @ qvec
            rank_arrays[key] = ranks_from_scores(scores)

        fused_scores = rrf_fuse(rank_arrays, UNIFORM_WEIGHTS)
        fused_ranks = ranks_from_scores(fused_scores)

        # I 100 candidati da rileggere col cross-encoder, in ordine di fusione.
        candidate_order = np.argsort(-fused_scores, kind="stable")[:RERANK_K]

        pairs = [(text, texts[idx]) for idx in candidate_order]
        rerank_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        # Nuovo ordine dei 100 candidati secondo il cross-encoder (punteggio più alto = più rilevante).
        new_order_within_candidates = np.argsort(-np.asarray(rerank_scores), kind="stable")
        reranked_candidate_idx = candidate_order[new_order_within_candidates]

        # Classifica finale: i 100 candidati riordinati dal reranker occupano i rank 1-100,
        # tutto il resto mantiene il rank che aveva già nella fusione (>100, invariato).
        final_ranks = fused_ranks.copy()
        for position, original_idx in enumerate(reranked_candidate_idx):
            final_ranks[original_idx] = position + 1

        gold_ranks_fused = [int(fused_ranks[i]) for i in gold_idx]
        gold_ranks_reranked = [int(final_ranks[i]) for i in gold_idx]

        row = {
            "query_id": query_id,
            "query_text": text,
            "n_gold": len(gold_ids),
            "fused": {
                "gold_ranks": gold_ranks_fused,
                **{f"recall@{k}": recall_at_k_from_ranks(gold_ranks_fused, k) for k in K_VALUES},
                "ndcg@10": ndcg_at_k_from_ranks(gold_ranks_fused, 10),
            },
            "reranked": {
                "gold_ranks": gold_ranks_reranked,
                **{f"recall@{k}": recall_at_k_from_ranks(gold_ranks_reranked, k) for k in K_VALUES},
                "ndcg@10": ndcg_at_k_from_ranks(gold_ranks_reranked, 10),
            },
        }
        per_query_results.append(row)
        print(f"\n{query_id}  ({text!r}, {row['n_gold']} oro)")
        print(f"  fuso (no rerank)   rank oro={sorted(gold_ranks_fused)}  recall@10={row['fused']['recall@10']:.2f}")
        print(f"  con reranking      rank oro={sorted(gold_ranks_reranked)}  recall@10={row['reranked']['recall@10']:.2f}")

    summary = {"fused": {}, "reranked": {}}
    for group in ("fused", "reranked"):
        for k in K_VALUES:
            summary[group][f"recall@{k}"] = float(np.mean([row[group][f"recall@{k}"] for row in per_query_results]))
        summary[group]["ndcg@10"] = float(np.mean([row[group]["ndcg@10"] for row in per_query_results]))

    print("\n=== Medie su 12 domande: fusione sola vs fusione + reranking ===")
    for group in ("fused", "reranked"):
        vals = summary[group]
        print(f"  {group:<10} " + "  ".join(f"@{k}={vals[f'recall@{k}']:.3f}" for k in K_VALUES) + f"  nDCG@10={vals['ndcg@10']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reranking_bge_large.json").write_text(
        json.dumps({
            "k_values": K_VALUES,
            "rerank_k": RERANK_K,
            "rrf_c": RRF_C,
            "weights": UNIFORM_WEIGHTS,
            "reranker": RERANKER_NAME,
            "per_query": per_query_results,
            "summary": summary,
        }, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(K_VALUES, [summary["fused"][f"recall@{k}"] for k in K_VALUES],
             marker="s", linestyle="--", color="gray", label="RRF uniforme, senza reranking (Fig6.1)")
    ax.plot(K_VALUES, [summary["reranked"][f"recall@{k}"] for k in K_VALUES],
             marker="o", color="tab:blue", label=f"RRF uniforme + {RERANKER_NAME} sui top-{RERANK_K}")
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel(f"k (dentro i {RERANK_K} candidati rerankati)")
    ax.set_ylabel("Recall@k medio su 12 domande")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Effetto del reranking sui top-100 fusi (corpus FiQA completo)")
    fig.tight_layout()
    out_path = OUT_DIR / "reranking_bge_large.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out_path}")


if __name__ == "__main__":
    main()
