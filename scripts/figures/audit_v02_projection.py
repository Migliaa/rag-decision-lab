"""Confronta vicinanza PCA 2D e ranking cosine originale per un caso V02."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval import BGE_QUERY_INSTRUCTION


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "runs/E001-20260912T141150Z-pilot512/figures/V02-V03-data.json"
CORPUS_PATH = ROOT / "data/processed/E001-pilot-512/corpus.jsonl"
MODEL_CACHE = ROOT / "data/cache/huggingface"
EMBEDDING_CACHE = ROOT / "data/cache/d1-lab-512"
QUERY_ID = "dc1aaac0b33553d8c897d4150955d803<::>7"
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", ""),
    "bge": ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", BGE_QUERY_INSTRUCTION),
}


def rank_positions(order: np.ndarray) -> dict[int, int]:
    return {int(row): position for position, row in enumerate(order, start=1)}


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    corpus = [json.loads(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()]
    ids = [row["_id"] for row in corpus]
    id_to_row = {passage_id: row for row, passage_id in enumerate(ids)}
    query = next(item for item in data["queries"] if item["id"] == QUERY_ID)
    report = {"query_id": QUERY_ID, "query": query["lastturn"], "models": {}}

    for method, (model_name, revision, instruction) in MODELS.items():
        model = SentenceTransformer(
            model_name,
            revision=revision,
            device="cpu",
            cache_folder=str(MODEL_CACHE),
            local_files_only=True,
        )
        query_vector = model.encode([instruction + query["lastturn"]], normalize_embeddings=True)[0]
        corpus_vectors = np.load(EMBEDDING_CACHE / f"{method}.npy")
        cosine_scores = corpus_vectors @ query_vector
        cosine_order = np.argsort(-cosine_scores, kind="stable")
        cosine_ranks = rank_positions(cosine_order)

        xy = np.asarray([[point["x"], point["y"]] for point in data["points"][method]])
        query_xy = np.asarray(query["coordinates"][method])
        projected_distances = np.linalg.norm(xy - query_xy, axis=1)
        projected_order = np.argsort(projected_distances, kind="stable")
        projected_ranks = rank_positions(projected_order)

        report["models"][method] = {
            "gold": [
                {
                    "id": passage_id,
                    "cosine_rank_384d": cosine_ranks[id_to_row[passage_id]],
                    "pca_distance_rank_2d": projected_ranks[id_to_row[passage_id]],
                }
                for passage_id in query["relevant_ids"]
            ],
            "closest_in_2d": [
                {
                    "id": ids[int(row)],
                    "pca_distance_rank_2d": projected_ranks[int(row)],
                    "cosine_rank_384d": cosine_ranks[int(row)],
                }
                for row in projected_order[:5]
            ],
        }

    output = DATA_PATH.with_name("V02-projection-audit.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
