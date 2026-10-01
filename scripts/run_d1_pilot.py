"""Esegue BM25, MiniLM e BGE sul medesimo pilot FiQA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import psutil

from src.retrieval import BGE_QUERY_INSTRUCTION, BM25Retriever, DenseRetriever, Passage


ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = ROOT / "data/cache/huggingface"
MODELS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "query_instruction": "",
    },
    "bge": {
        "name": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "query_instruction": BGE_QUERY_INSTRUCTION,
    },
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=int, default=512)
    return parser.parse_args()


def load_qrels(pilot_dir: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with (pilot_dir / "qrels.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            qrels[row["query-id"]].add(row["corpus-id"])
    return qrels


def recall_at(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def ndcg_at(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(rank + 2) for rank, pid in enumerate(ranked_ids[:k]) if pid in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def truncation_stats(retriever: DenseRetriever, texts: list[str]) -> dict:
    tokenizer = retriever.model.tokenizer
    max_tokens = retriever.model.max_seq_length
    lengths = [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts]
    return {
        "max_sequence_length": max_tokens,
        "texts_over_limit": sum(length > max_tokens for length in lengths),
        "max_observed_tokens": max(lengths),
    }


def main() -> None:
    args = parse_args()
    pilot_dir = ROOT / f"data/processed/E001-pilot-{args.passages}"
    data_manifest_path = ROOT / f"data/manifests/E001-pilot-{args.passages}.json"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / f"E001-{timestamp}-pilot{args.passages}"
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    rss_samples: list[int] = []

    def event(kind: str, **details: object) -> None:
        if "rss_bytes" in details:
            rss_samples.append(int(details["rss_bytes"]))
        record = {"timestamp": datetime.now(UTC).isoformat(), "event": kind, **details}
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    passages = [Passage(row["_id"], row.get("title", ""), row["text"]) for row in read_jsonl(pilot_dir / "corpus.jsonl")]
    queries = read_jsonl(pilot_dir / "queries.jsonl")
    qrels = load_qrels(pilot_dir)
    process = psutil.Process()
    manifest = {
        "status": "running",
        "timestamp_utc": timestamp,
        "experiment_id": "E001",
        "purpose": "D1 pipeline and CPU pilot; scores are not full-corpus FiQA results",
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ["numpy", "rank-bm25", "sentence-transformers", "torch"]},
        "source_hashes": {
            "retrieval.py": source_sha256(ROOT / "src/retrieval.py"),
            "run_d1_pilot.py": source_sha256(Path(__file__)),
        },
        "data_manifest": str(data_manifest_path.relative_to(ROOT)),
        "data_manifest_sha256": source_sha256(data_manifest_path),
        "query_count": len(queries),
        "passage_count": len(passages),
        "top_k": 10,
        "models": MODELS,
        "methods": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    event("run_started", rss_bytes=process.memory_info().rss)

    retrievers: dict[str, object] = {}
    started = perf_counter()
    retrievers["bm25"] = BM25Retriever(passages)
    manifest["methods"]["bm25"] = {"index_seconds": perf_counter() - started}
    event("method_ready", method="bm25", rss_bytes=process.memory_info().rss)

    for method, config in MODELS.items():
        event("model_loading", method=method, model=config["name"], revision=config["revision"])
        retriever = DenseRetriever(
            passages,
            model_name=config["name"],
            revision=config["revision"],
            query_instruction=config["query_instruction"],
            batch_size=16,
            cache_folder=str(MODEL_CACHE),
        )
        retrievers[method] = retriever
        manifest["methods"][method] = {
            "corpus_encoding_seconds": retriever.encoding_seconds,
            "truncation": truncation_stats(retriever, [p.indexed_text for p in passages]),
        }
        event("method_ready", method=method, rss_bytes=process.memory_info().rss)

    detailed = []
    aggregates = []
    for method, retriever in retrievers.items():
        method_rows = []
        for query in queries:
            started = perf_counter()
            ranking = retriever.search(query["text"], top_k=10)
            search_seconds = perf_counter() - started
            ranked_ids = [item.passage_id for item in ranking]
            relevant = qrels[query["_id"]]
            row = {
                "method": method,
                "query_id": query["_id"],
                "query": query["text"],
                "relevant_ids": sorted(relevant),
                "ranking": [{"passage_id": item.passage_id, "score": item.score} for item in ranking],
                "recall_at_5": recall_at(ranked_ids, relevant, 5),
                "recall_at_10": recall_at(ranked_ids, relevant, 10),
                "ndcg_at_10": ndcg_at(ranked_ids, relevant, 10),
                "search_seconds": search_seconds,
            }
            detailed.append(row)
            method_rows.append(row)
        aggregates.append({
            "method": method,
            "query_count": len(method_rows),
            "recall_at_5": sum(row["recall_at_5"] for row in method_rows) / len(method_rows),
            "recall_at_10": sum(row["recall_at_10"] for row in method_rows) / len(method_rows),
            "ndcg_at_10": sum(row["ndcg_at_10"] for row in method_rows) / len(method_rows),
            "mean_search_seconds": sum(row["search_seconds"] for row in method_rows) / len(method_rows),
        })

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in detailed:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)

    manifest["status"] = "completed"
    manifest["peak_rss_bytes_observed"] = max(rss_samples)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    event("run_completed", rss_bytes=process.memory_info().rss)
    print(run_dir)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
