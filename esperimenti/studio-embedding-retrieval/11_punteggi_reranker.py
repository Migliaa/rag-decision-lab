"""Step 11: calcolare una volta sola i punteggi di piu' reranker sui candidati, e salvarli.

Separare il calcolo costoso dall'esplorazione delle strategie: qui si paga una
volta il cross-encoder su (domanda, candidato), i punteggi finiscono su disco, e
lo step 12 puo' poi confrontare decine di strategie di combinazione a costo zero.

Tre reranker gratuiti di taglia molto diversa, per poterli anche fondere tra loro:
- cross-encoder/ms-marco-MiniLM-L-6-v2 (22M parametri, addestrato su MS MARCO)
- BAAI/bge-reranker-large (560M, quello gia' usato nello step 9)
- BAAI/bge-reranker-v2-m3 (568M, successore dichiarato di bge-reranker-large)

Riprende da dove si era interrotto se il processo viene ucciso: salva i punteggi
dopo ogni domanda.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/11_punteggi_reranker.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

RRF_C = 60
N_CANDIDATES = 200  # lista larga: lo step 12 puo' poi valutare tagli piu' stretti gratis
FUSION_WEIGHTS = {"bm25": 1 / 3, "bge": 1 / 3, "minilm": 1 / 3}

MODELS = {
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2",
                   revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                   query_prefix="", embeddings_file="corpus_embeddings_minilm_full.npy"),
    "bge": dict(name="BAAI/bge-small-en-v1.5",
                revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                query_prefix="Represent this sentence for searching relevant passages: ",
                embeddings_file="corpus_embeddings_bge_full.npy"),
}

RERANKERS = {
    "ms-marco-MiniLM-L6": dict(name="cross-encoder/ms-marco-MiniLM-L-6-v2", depth=N_CANDIDATES, batch=32),
    "bge-reranker-large": dict(name="BAAI/bge-reranker-large",
                               revision="55611d7bca2a7133960a6d3b71e083071bbfc312", depth=100, batch=16),
    "bge-reranker-v2-m3": dict(name="BAAI/bge-reranker-v2-m3", depth=100, batch=16),
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


def build_candidates(ids, texts, query_ids, query_texts, gold_by_query, id_to_idx) -> dict:
    """Lista di candidati e rank di prima fase, salvata una volta per tutte."""
    cand_path = OUT_DIR / "candidati_180.npz"
    if cand_path.exists():
        print("Candidati gia' calcolati, li riuso.")
        data = np.load(cand_path, allow_pickle=True)
        return {"candidates": data["candidates"], "fusion_ranks": data["fusion_ranks"],
                "gold_ranks_full": data["gold_ranks_full"], "query_ids": list(data["query_ids"])}

    print("Indice BM25...")
    bm25 = BM25Okapi([lexical_tokens(t) for t in texts])
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

    candidates = np.zeros((len(query_ids), N_CANDIDATES), dtype=np.int64)
    fusion_ranks = np.zeros((len(query_ids), N_CANDIDATES), dtype=np.int64)
    gold_ranks_full = []

    for i, query_id in enumerate(query_ids):
        text = query_texts[query_id]
        rank_arrays = {"bm25": ranks_from_scores(np.asarray(bm25.get_scores(lexical_tokens(text))))}
        for key in MODELS:
            rank_arrays[key] = ranks_from_scores(corpus_embeddings[key] @ query_vectors[key][i])
        fused = np.zeros(len(ids), dtype=np.float64)
        for method, ranks in rank_arrays.items():
            fused += FUSION_WEIGHTS[method] * (1.0 / (RRF_C + ranks))
        order = np.argsort(-fused, kind="stable")[:N_CANDIDATES]
        candidates[i] = order
        fusion_ranks[i] = np.arange(1, N_CANDIDATES + 1)
        full_ranks = ranks_from_scores(fused)
        gold_ranks_full.append([int(full_ranks[id_to_idx[g]]) for g in gold_by_query[query_id] if g in id_to_idx])
        if (i + 1) % 20 == 0:
            print(f"  candidati {i + 1}/{len(query_ids)}")

    np.savez(cand_path, candidates=candidates, fusion_ranks=fusion_ranks,
             gold_ranks_full=np.array(gold_ranks_full, dtype=object), query_ids=np.array(query_ids))
    return {"candidates": candidates, "fusion_ranks": fusion_ranks,
            "gold_ranks_full": gold_ranks_full, "query_ids": query_ids}


def score_with_reranker(key: str, cfg: dict, query_ids, query_texts, candidates, texts) -> None:
    """Punteggi del cross-encoder su ogni (domanda, candidato), con ripresa dopo interruzione."""
    depth = cfg["depth"]
    scores_path = OUT_DIR / f"rerank_scores_{key}.npy"
    progress_path = OUT_DIR / f"rerank_progress_{key}.json"

    if scores_path.exists() and progress_path.exists():
        scores = np.load(scores_path)
        done = json.loads(progress_path.read_text())["done"]
        if done >= len(query_ids):
            print(f"{key}: gia' completo.")
            return
        print(f"{key}: riprendo da {done}/{len(query_ids)}")
    else:
        scores = np.full((len(query_ids), depth), np.nan, dtype=np.float32)
        done = 0

    model_kwargs = {"revision": cfg["revision"]} if "revision" in cfg else {}
    model = CrossEncoder(cfg["name"], device="cpu", cache_folder=str(CACHE_HF), max_length=512, **model_kwargs)

    start = time.time()
    for i in range(done, len(query_ids)):
        q = query_texts[query_ids[i]]
        pairs = [(q, texts[idx]) for idx in candidates[i][:depth]]
        scores[i] = np.asarray(model.predict(pairs, batch_size=cfg["batch"], show_progress_bar=False), dtype=np.float32)
        np.save(scores_path, scores)
        progress_path.write_text(json.dumps({"done": i + 1, "total": len(query_ids), "depth": depth}))
        if (i + 1) % 10 == 0 or i == done:
            elapsed = time.time() - start
            per_q = elapsed / (i - done + 1)
            left = per_q * (len(query_ids) - i - 1)
            print(f"  {key}: {i + 1}/{len(query_ids)}  ({per_q:.1f} s/domanda, ~{left / 60:.0f} min rimanenti)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Carico il corpus...")
    ids, texts = load_corpus(CORPUS_PATH)
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    query_texts = load_query_texts(set(query_ids), MTRAG / "fiqa_rewrite.jsonl")
    print(f"Domande: {len(query_ids)}   passaggi: {len(ids)}")

    cand = build_candidates(ids, texts, query_ids, query_texts, gold_by_query, id_to_idx)
    candidates = cand["candidates"]

    # Quanto oro c'e' davvero dentro la lista di candidati: e' il tetto invalicabile
    # per qualunque strategia di reranking costruita su questa lista.
    for depth in (50, 100, 200):
        got = np.mean([
            sum(1 for r in gr if r <= depth) / len(gr) for gr in cand["gold_ranks_full"]
        ])
        print(f"  tetto di recall con {depth} candidati: {got:.3f}")

    for key, cfg in RERANKERS.items():
        print(f"\n=== {key} ({cfg['name']}) ===")
        score_with_reranker(key, cfg, query_ids, query_texts, candidates, texts)

    print("\nPunteggi salvati. Lo step 12 puo' ora confrontare le strategie senza ricalcolare nulla.")


if __name__ == "__main__":
    main()
