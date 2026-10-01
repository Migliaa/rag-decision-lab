"""Prepara embedding, PCA e dati V02/V03 dalla run E001 a 512 passaggi."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from src.retrieval import BGE_QUERY_INSTRUCTION, DenseRetriever, Passage


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "data/processed/E001-pilot-512"
RUN_DIR = ROOT / "runs/E001-20260912T141150Z-pilot512"
MODEL_CACHE = ROOT / "data/cache/huggingface"
EMBEDDING_CACHE = ROOT / "data/cache/d1-lab-512"
OUTPUT = RUN_DIR / "figures/V02-V03-data.json"
MODELS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "instruction": "",
    },
    "bge": {
        "name": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "instruction": BGE_QUERY_INSTRUCTION,
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    corpus_rows = read_jsonl(PILOT_DIR / "corpus.jsonl")
    passages = [Passage(row["_id"], row.get("title", ""), row["text"]) for row in corpus_rows]
    queries = read_jsonl(PILOT_DIR / "queries.jsonl")
    variants = {}
    for filename, key in [
        ("fiqa_questions.jsonl", "history"),
        ("fiqa_rewrite.jsonl", "rewrite"),
    ]:
        path = ROOT / "data/raw/mtrag-source/mtrag-human/retrieval_tasks/fiqa" / filename
        variants[key] = {row["_id"]: row["text"] for row in read_jsonl(path)}

    run_rows = read_jsonl(RUN_DIR / "results.jsonl")
    by_query_method = {(row["query_id"], row["method"]): row for row in run_rows}
    relevant_by_query = {row["query_id"]: row["relevant_ids"] for row in run_rows if row["method"] == "bm25"}
    passage_by_id = {row["_id"]: row for row in corpus_rows}

    visual = {"points": {}, "queries": [], "projection": {}}
    EMBEDDING_CACHE.mkdir(parents=True, exist_ok=True)
    for method, config in MODELS.items():
        retriever = DenseRetriever(
            passages,
            config["name"],
            config["revision"],
            query_instruction=config["instruction"],
            batch_size=16,
            cache_folder=str(MODEL_CACHE),
        )
        np.save(EMBEDDING_CACHE / f"{method}.npy", retriever.embeddings)
        reducer = PCA(n_components=2, random_state=42)
        coordinates = reducer.fit_transform(retriever.embeddings)
        visual["projection"][method] = {
            "method": "PCA",
            "explained_variance_ratio": [float(value) for value in reducer.explained_variance_ratio_],
        }
        visual["points"][method] = [
            {
                "id": passage.passage_id,
                "x": round(float(coordinates[index, 0]), 6),
                "y": round(float(coordinates[index, 1]), 6),
                "text": " ".join(passage.indexed_text.split())[:360],
            }
            for index, passage in enumerate(passages)
        ]
        for query in queries:
            vector = retriever.model.encode(
                [config["instruction"] + query["text"]],
                normalize_embeddings=True,
            )
            query.setdefault("coordinates", {})[method] = [round(float(value), 6) for value in reducer.transform(vector)[0]]

    for query in queries:
        item = {
            "id": query["_id"],
            "lastturn": query["text"],
            "history": variants["history"].get(query["_id"], ""),
            "rewrite": variants["rewrite"].get(query["_id"], ""),
            "coordinates": query["coordinates"],
            "relevant_ids": relevant_by_query[query["_id"]],
            "rankings": {},
        }
        for method in ["bm25", "minilm", "bge"]:
            row = by_query_method[(query["_id"], method)]
            item["rankings"][method] = [
                {
                    "rank": index + 1,
                    "id": result["passage_id"],
                    "score": round(float(result["score"]), 5),
                    "relevant": result["passage_id"] in item["relevant_ids"],
                    "text": " ".join(passage_by_id[result["passage_id"]]["text"].split())[:280],
                }
                for index, result in enumerate(row["ranking"][:5])
            ]
        visual["queries"].append(item)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(visual, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
