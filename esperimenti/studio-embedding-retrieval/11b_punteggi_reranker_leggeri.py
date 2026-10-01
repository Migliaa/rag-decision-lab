"""Step 11b: punteggi di due reranker leggeri in piu', per poterli fondere tra loro.

Copiato da 11_punteggi_reranker.py, cambia solo l'insieme dei modelli.

Motivo del cambio: bge-reranker-large, il modello piu' forte fra quelli gratuiti, su
questa CPU impiega circa 2,7 minuti per domanda con 100 candidati - otto ore per una
sola valutazione su 180 domande. Non e' un limite del modello ma dell'hardware
disponibile, e va registrato come vincolo di design: il reranker migliore e'
inutilizzabile senza GPU. Al suo posto, due cross-encoder piccoli scelti in modo che
non vengano dalla stessa famiglia, perche' fondere due modelli che sbagliano nello
stesso modo non aggiunge informazione.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/11b_punteggi_reranker_leggeri.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

RERANKERS = {
    # stessa famiglia del primo ma piu' profondo: serve a capire se basta piu' capacita'
    "ms-marco-MiniLM-L12": dict(name="cross-encoder/ms-marco-MiniLM-L-12-v2", depth=100, batch=32),
    # famiglia diversa (DeBERTa-v3 invece di BERT/MiniLM): errori piu' indipendenti
    "mxbai-rerank-xsmall": dict(name="mixedbread-ai/mxbai-rerank-xsmall-v1", depth=100, batch=32),
}


def indexed_text(passage: dict) -> str:
    title = passage.get("title", "")
    text = passage["text"]
    return f"{title}\n{text}" if title else text


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


def score_with_reranker(key: str, cfg: dict, query_ids, query_texts, candidates, texts) -> None:
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

    model = CrossEncoder(cfg["name"], device="cpu", cache_folder=str(CACHE_HF), max_length=512)

    start = time.time()
    for i in range(done, len(query_ids)):
        q = query_texts[query_ids[i]]
        pairs = [(q, texts[idx]) for idx in candidates[i][:depth]]
        scores[i] = np.asarray(model.predict(pairs, batch_size=cfg["batch"], show_progress_bar=False), dtype=np.float32)
        np.save(scores_path, scores)
        progress_path.write_text(json.dumps({"done": i + 1, "total": len(query_ids), "depth": depth}))
        if (i + 1) % 10 == 0 or i == done:
            per_q = (time.time() - start) / (i - done + 1)
            print(f"  {key}: {i + 1}/{len(query_ids)}  ({per_q:.1f} s/domanda, "
                  f"~{per_q * (len(query_ids) - i - 1) / 60:.0f} min rimanenti)", flush=True)


def main() -> None:
    ids, texts = load_corpus(CORPUS_PATH)
    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    query_texts = load_query_texts(set(query_ids), MTRAG / "fiqa_rewrite.jsonl")

    data = np.load(OUT_DIR / "candidati_180.npz", allow_pickle=True)
    candidates = data["candidates"]
    print(f"Domande: {len(query_ids)}   candidati per domanda: {candidates.shape[1]}")

    for key, cfg in RERANKERS.items():
        print(f"\n=== {key} ({cfg['name']}) ===", flush=True)
        score_with_reranker(key, cfg, query_ids, query_texts, candidates, texts)


if __name__ == "__main__":
    main()
