"""Congela un pilot D1 riproducibile senza scegliere casi dai risultati."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data/raw/fiqa/fiqa.jsonl"
QUERY_PATH = ROOT / "data/raw/mtrag-source/mtrag-human/retrieval_tasks/fiqa/fiqa_lastturn.jsonl"
QRELS_PATH = ROOT / "data/raw/mtrag-source/mtrag-human/retrieval_tasks/fiqa/qrels/dev.tsv"
MTRAG_REVISION = "2c618bb98db3c8526433e22d8a2f7320f10a7470"
PILOT_SEED = "E001-D1-fiqa-pilot-v1"
QUERY_COUNT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=int, default=512)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_key(value: str) -> str:
    # Non usiamo random senza traccia: dalla coppia seed+ID otteniamo sempre
    # lo stesso valore. Ordinare per questo valore congela la selezione prima
    # di conoscere i risultati dei retriever.
    return hashlib.sha256(f"{PILOT_SEED}:{value}".encode()).hexdigest()


def main() -> None:
    args = parse_args()
    passage_count = args.passages
    output_dir = ROOT / f"data/processed/E001-pilot-{passage_count}"
    manifest_path = ROOT / f"data/manifests/E001-pilot-{passage_count}.json"
    # A) Leggiamo le query ufficiali "ultimo turno". Non le generiamo noi.
    queries = read_jsonl(QUERY_PATH)
    selected_queries = sorted(queries, key=lambda row: stable_key(row["_id"]))[:QUERY_COUNT]
    selected_query_ids = {row["_id"] for row in selected_queries}

    # B) Le qrels sono la risposta del valutatore: query_id -> ID pertinenti.
    # Servono per comporre e valutare il pilot, mai come input del ranking.
    qrels: dict[str, set[str]] = defaultdict(set)
    with QRELS_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["query-id"] in selected_query_ids and int(row["score"]) > 0:
                qrels[row["query-id"]].add(row["corpus-id"])

    # C) Conserviamo TUTTE le fonti annotate per le 12 query selezionate.
    gold_ids = set().union(*(qrels[qid] for qid in selected_query_ids))
    if len(gold_ids) >= passage_count:
        raise ValueError("Il numero di fonti pertinenti non lascia spazio ai distrattori")

    corpus = read_jsonl(CORPUS_PATH)
    corpus_by_id = {row["_id"]: row for row in corpus}
    missing = gold_ids - corpus_by_id.keys()
    if missing:
        raise ValueError(f"Qrels con ID assenti dal corpus: {sorted(missing)[:5]}")

    # D) Riempiamo il campione con passaggi non-gold scelti senza guardare
    # punteggi o contenuti. Sono i documenti che il retriever deve scartare.
    distractors = sorted(
        (row for row in corpus if row["_id"] not in gold_ids),
        key=lambda row: stable_key(row["_id"]),
    )[: passage_count - len(gold_ids)]
    selected_passages = sorted(
        [corpus_by_id[passage_id] for passage_id in gold_ids] + distractors,
        key=lambda row: row["_id"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    # E) Scriviamo file nuovi sotto data/processed; gli originali in raw
    # restano immutati e sono sempre recuperabili tramite revisione + hash.
    with (output_dir / "corpus.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected_passages:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as stream:
        for row in sorted(selected_queries, key=lambda item: item["_id"]):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "qrels.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for query_id in sorted(qrels):
            for passage_id in sorted(qrels[query_id]):
                writer.writerow([query_id, passage_id, 1])

    manifest = {
        "dataset": "MTRAG Human / FiQA passage-level",
        "source_repository": "https://github.com/IBM/mt-rag-benchmark",
        "source_revision": MTRAG_REVISION,
        "reported_passage_count_in_readme": 49607,
        "observed_passage_count": len(corpus),
        "observed_query_count_lastturn": len(queries),
        "inputs": {
            "corpus_zip_sha256": "59f7cf5b043abe007fdb6521eb281bf055524de4376466e9cb4808466e761ee9",
            "corpus_jsonl_sha256": sha256(CORPUS_PATH),
            "queries_sha256": sha256(QUERY_PATH),
            "qrels_sha256": sha256(QRELS_PATH),
        },
        "pilot": {
            "purpose": "pipeline and CPU pilot; not an official full-corpus FiQA score",
            "selection": "SHA-256 ordering by fixed seed; all gold passages retained, remainder deterministic distractors",
            "seed": PILOT_SEED,
            "query_count": len(selected_queries),
            "passage_count": len(selected_passages),
            "gold_passage_count": len(gold_ids),
            "query_ids": sorted(selected_query_ids),
            "corpus_sha256": sha256(output_dir / "corpus.jsonl"),
            "queries_sha256": sha256(output_dir / "queries.jsonl"),
            "qrels_sha256": sha256(output_dir / "qrels.tsv"),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["pilot"], indent=2))


if __name__ == "__main__":
    main()
