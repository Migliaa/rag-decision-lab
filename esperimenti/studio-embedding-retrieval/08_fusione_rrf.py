"""Step 8: fondere BM25+MiniLM+BGE con RRF pesato, e misurare quanto in
profondità bisogna guardare per catturare l'oro nella lista fusa.

Non richiede nuovi embedding: usa le matrici già calcolate e salvate in
07_metriche_corpus_completo.py (corpus_embeddings_*_full.npy) sul corpus
FiQA completo. Solo le 12 query vengono incorporate qui (12 chiamate al
modello, non 61.022).

Due domande in un colpo solo:
1. Tra tre schemi di pesi diversi per RRF, quale copre meglio l'oro?
2. Nella classifica fusa (migliore schema o no), quanto in basso arriva
   l'oro nei casi peggiori? È il numero che serve per dimensionare il
   reranking del prossimo step — non un'ipotesi a tavolino.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/08_fusione_rrf.py
"""

from __future__ import annotations

import json
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
PILOT_QRELS_PATH = REPO_ROOT / "data" / "processed" / "E001-pilot-512" / "qrels.tsv"
REWRITE_PATH = (
    REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa" / "fiqa_rewrite.jsonl"
)
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

RRF_C = 60  # costante standard, vedi Appunti4
K_VALUES = [10, 50, 100, 500]  # a quali profondità misurare il recall della lista fusa

# I tre schemi discussi con Andrea: la sua proposta, l'inverso (coerente con
# Fig3.2, dove MiniLM ha battuto BGE a scala piena), e uniforme come riferimento neutro.
WEIGHT_SCHEMES = {
    "proposto (BM25 20 / BGE 50 / MiniLM 30)": {"bm25": 0.20, "bge": 0.50, "minilm": 0.30},
    "invertito (BM25 20 / BGE 30 / MiniLM 50)": {"bm25": 0.20, "bge": 0.30, "minilm": 0.50},
    "uniforme (33 / 33 / 33)": {"bm25": 1 / 3, "bge": 1 / 3, "minilm": 1 / 3},
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
    """Da punteggi grezzi a rank 1-indicizzato per ciascun indice originale.

    ranks[i] = posizione (1 = il migliore) del passaggio i-esimo in questa
    classifica. Vettorizzato: invertire l'ordinamento è più rapido che
    costruire un dizionario id->rank su 61.022 elementi per ogni query.
    """
    order = np.argsort(-scores, kind="stable")  # order[posizione] = indice originale
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def rrf_fuse(rank_arrays: dict[str, np.ndarray], weights: dict[str, float], c: int = RRF_C) -> np.ndarray:
    """RRF pesato, calcolato per tutti gli indici in un colpo solo (numpy)."""
    n = len(next(iter(rank_arrays.values())))
    fused = np.zeros(n, dtype=np.float64)
    for method, ranks in rank_arrays.items():
        fused += weights[method] * (1.0 / (c + ranks))
    return fused


def recall_at_k_from_ranks(gold_ranks: list[int], k: int) -> float:
    return sum(1 for r in gold_ranks if r <= k) / len(gold_ranks)


def ndcg_at_k_from_ranks(gold_ranks: list[int], k: int) -> float:
    import math
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

    # id -> indice originale, per tradurre gli id oro in posizioni nell'array
    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    per_query_results = []  # per ogni domanda: per ogni schema, i rank oro nella lista fusa

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

        row = {"query_id": query_id, "query_text": text, "n_gold": len(gold_ids), "schemes": {}}
        for scheme_name, weights in WEIGHT_SCHEMES.items():
            fused_scores = rrf_fuse(rank_arrays, weights)
            fused_ranks = ranks_from_scores(fused_scores)
            gold_ranks_fused = [int(fused_ranks[i]) for i in gold_idx]
            row["schemes"][scheme_name] = {
                "gold_ranks": gold_ranks_fused,
                **{f"recall@{k}": recall_at_k_from_ranks(gold_ranks_fused, k) for k in K_VALUES},
                "ndcg@10": ndcg_at_k_from_ranks(gold_ranks_fused, 10),
            }

        # riferimento: MiniLM da solo (il migliore dei tre singoli in Fig3.2), non fuso
        minilm_gold_ranks = [int(rank_arrays["minilm"][i]) for i in gold_idx]
        row["minilm_solo"] = {
            "gold_ranks": minilm_gold_ranks,
            **{f"recall@{k}": recall_at_k_from_ranks(minilm_gold_ranks, k) for k in K_VALUES},
        }

        per_query_results.append(row)
        print(f"\n{query_id}  ({text!r}, {row['n_gold']} oro)")
        for scheme_name in WEIGHT_SCHEMES:
            r = row["schemes"][scheme_name]
            print(f"  {scheme_name:<42} rank oro={sorted(r['gold_ranks'])}  recall@10={r['recall@10']:.2f}")

    # aggregati
    summary = {}
    for scheme_name in WEIGHT_SCHEMES:
        summary[scheme_name] = {
            f"recall@{k}": float(np.mean([row["schemes"][scheme_name][f"recall@{k}"] for row in per_query_results]))
            for k in K_VALUES
        }
        summary[scheme_name]["ndcg@10"] = float(np.mean([row["schemes"][scheme_name]["ndcg@10"] for row in per_query_results]))
    summary["minilm_solo"] = {
        f"recall@{k}": float(np.mean([row["minilm_solo"][f"recall@{k}"] for row in per_query_results]))
        for k in K_VALUES
    }

    all_deepest_gold_ranks = [
        max(row["schemes"][scheme_name]["gold_ranks"])
        for row in per_query_results
        for scheme_name in WEIGHT_SCHEMES
    ]
    print("\n=== Medie su 12 domande (recall@k per schema di pesi) ===")
    for scheme_name, vals in summary.items():
        print(f"  {scheme_name:<42} " + "  ".join(f"@{k}={vals[f'recall@{k}']:.3f}" for k in K_VALUES))

    print("\n=== Profondità dell'oro nella lista fusa (tutti gli schemi insieme) ===")
    arr = np.array(all_deepest_gold_ranks)
    print(f"  mediana={np.median(arr):.0f}  90° percentile={np.percentile(arr, 90):.0f}  massimo={arr.max():.0f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fusione_rrf.json").write_text(
        json.dumps({"k_values": K_VALUES, "rrf_c": RRF_C, "weight_schemes": WEIGHT_SCHEMES,
                    "per_query": per_query_results, "summary": summary}, indent=2),
        encoding="utf-8",
    )

    # Figura: curve di recall@k (asse x = k, log) per i tre schemi + riferimento MiniLM solo
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for scheme_name in WEIGHT_SCHEMES:
        ax.plot(K_VALUES, [summary[scheme_name][f"recall@{k}"] for k in K_VALUES], marker="o", label=scheme_name)
    ax.plot(K_VALUES, [summary["minilm_solo"][f"recall@{k}"] for k in K_VALUES],
            marker="s", linestyle="--", color="gray", label="MiniLM da solo (riferimento, non fuso)")
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("k (quanti risultati si guardano nella lista fusa)")
    ax.set_ylabel("Recall@k medio su 12 domande")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Corpus FiQA completo (61.022 passaggi) — RRF pesato, tre schemi")
    fig.tight_layout()
    out_path = OUT_DIR / "fusione_rrf.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out_path}")


if __name__ == "__main__":
    main()
