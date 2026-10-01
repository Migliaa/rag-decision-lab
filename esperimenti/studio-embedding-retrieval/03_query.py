"""Step 3: dalla domanda alla classifica, con controllo contro le qrels.

Riusa le matrici già salvate (01_indicizzazione.py, 01b_indicizzazione_bge.py):
non ricalcola l'embedding del corpus, incorpora solo la query e la confronta.

Una sola domanda per ora, scelta perché ha oro noto nelle qrels del pilota:
possiamo dire non solo "cosa trova il sistema" ma "lo trova giusto o no".
Un solo esempio mostra il meccanismo, non misura la qualità del metodo — il
recall@k su una domanda sola non si generalizza (step successivo: farlo su
tutte le 12 domande del pilota).

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/03_query.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "E001-pilot-512" / "corpus.jsonl"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

TOP_K = 10

# Turno 3 della conversazione 2f94b6fb...: riscrittura auto-contenuta, non
# l'ultima frase isolata (vedi conversazione su lastturn/rewrite/questions).
QUERY_ID = "2f94b6fb4c6f941decd0609eb126610b<::>3"
QUERY_TEXT = "What is an IRA?"
GOLD_IDS = {"448260-0-1360", "532839-0-712", "94496-0-2276"}  # da data/processed/E001-pilot-512/qrels.tsv

MODELS = {
    "minilm": dict(
        name="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        query_prefix="",
        embeddings_file="corpus_embeddings_minilm.npy",
    ),
    "bge": dict(
        name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        # Qui, a differenza dell'indicizzazione del corpus, il prefisso serve:
        # la query è per definizione "una domanda che cerca", il passaggio no.
        query_prefix="Represent this sentence for searching relevant passages: ",
        embeddings_file="corpus_embeddings_bge.npy",
    ),
}


def load_corpus_texts(path: Path) -> dict[str, str]:
    texts = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts[row["_id"]] = row["text"]
    return texts


def rank_for_model(model_key: str, ids: list[str]) -> list[dict]:
    cfg = MODELS[model_key]
    corpus_embeddings = np.load(OUT_DIR / cfg["embeddings_file"])

    model = SentenceTransformer(cfg["name"], revision=cfg["revision"], device="cpu", cache_folder=str(CACHE_HF))
    query_vector = model.encode([cfg["query_prefix"] + QUERY_TEXT], normalize_embeddings=True)[0]

    # Stesso identico calcolo dello step concettuale spiegato prima:
    # prodotto scalare tra la query e OGNI riga della matrice del corpus.
    scores = corpus_embeddings @ query_vector
    order = np.argsort(-scores, kind="stable")[:TOP_K]

    return [{"rank": rank + 1, "id": ids[i], "score": float(scores[i])} for rank, i in enumerate(order)]


def main() -> None:
    ids = json.loads((OUT_DIR / "corpus_ids.json").read_text(encoding="utf-8"))
    texts = load_corpus_texts(CORPUS_PATH)

    print(f"Domanda: {QUERY_TEXT!r}  (id qrels: {QUERY_ID})")
    print(f"Oro atteso ({len(GOLD_IDS)} passaggi): {sorted(GOLD_IDS)}\n")

    results_to_save = {"query_id": QUERY_ID, "query_text": QUERY_TEXT, "gold_ids": sorted(GOLD_IDS), "models": {}}

    for model_key in MODELS:
        print(f"--- {model_key} ---")
        ranking = rank_for_model(model_key, ids)

        found_gold = set()
        for row in ranking:
            hit = row["id"] in GOLD_IDS
            if hit:
                found_gold.add(row["id"])
            marker = "[ORO]" if hit else "     "
            preview = texts[row["id"]][:80].replace("\n", " ")
            print(f"  {row['rank']:>2}. {marker} score={row['score']:.3f}  {row['id']:<20} {preview}")

        recall_at_k = len(found_gold) / len(GOLD_IDS)
        missed = GOLD_IDS - found_gold
        print(f"  Recall@{TOP_K} per QUESTA SOLA domanda: {recall_at_k:.0%} ({len(found_gold)}/{len(GOLD_IDS)})")
        if missed:
            print(f"  Oro non trovato in top-{TOP_K}: {sorted(missed)}")
        print()

        results_to_save["models"][model_key] = {"ranking": ranking, "recall_at_k": recall_at_k}

    out_path = OUT_DIR / "query_ranking_example.json"
    out_path.write_text(json.dumps(results_to_save, indent=2), encoding="utf-8")
    print(f"Risultati salvati in: {out_path}")


if __name__ == "__main__":
    main()
