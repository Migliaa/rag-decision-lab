"""Inserisci una domanda e confronta BM25, MiniLM e BGE.

Questo file è intenzionalmente lineare e commentato. Il codice più generico che
prepara manifest e run resta in ``scripts/``; qui guardiamo solo le decisioni
che cambiano il comportamento della ricerca.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval import BGE_QUERY_INSTRUCTION, BM25Retriever, Passage


# 1) DOVE SONO I DATI PROCESSATI?
# Usiamo il pilot da 512 passaggi. Non tocchiamo il corpus originale scaricato.
ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "data/processed/E001-pilot-512"
MODEL_CACHE = ROOT / "data/cache/huggingface"
EMBEDDING_CACHE = ROOT / "data/cache/d1-lab-512"
TOP_K = 5

# 2) QUALI MODELLI USIAMO?
# Nome e revisione devono combaciare con quelli usati per creare gli embedding
# salvati. BGE riceve anche l'istruzione; MiniLM no.
MODELS = {
    "MiniLM": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "instruction": "",
        "cache_file": "minilm.npy",
    },
    "BGE": {
        "name": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "instruction": BGE_QUERY_INSTRUCTION,
        "cache_file": "bge.npy",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    """Legge un record JSON per riga e restituisce una lista di dizionari."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def short(text: str, length: int = 240) -> str:
    """Compatta un passaggio soltanto per stamparlo; il ranking usa il testo intero."""
    compact = " ".join(text.split())
    return compact if len(compact) <= length else compact[: length - 1] + "…"


def show_ranking(name: str, ranking: list[tuple[str, float]], by_id: dict[str, Passage]) -> None:
    """Mostra ID, punteggio e testo: un numero senza fonte non è interpretabile."""
    print(f"\n=== {name} ===")
    for position, (passage_id, score) in enumerate(ranking, start=1):
        print(f"{position}. {passage_id}  score={score:.4f}")
        print(f"   {short(by_id[passage_id].indexed_text)}")


def main() -> None:
    # 3) PREPARAZIONE UNA TANTUM.
    # Corpus, indice BM25, modelli e vettori vengono caricati una volta. In questo
    # modo Andrea può provare più domande senza ripetere l'indicizzazione.
    rows = read_jsonl(PILOT_DIR / "corpus.jsonl")
    passages = [Passage(row["_id"], row.get("title", ""), row["text"]) for row in rows]
    by_id = {passage.passage_id: passage for passage in passages}
    bm25 = BM25Retriever(passages)
    dense = {}
    for label, config in MODELS.items():
        model = SentenceTransformer(
            config["name"],
            revision=config["revision"],
            device="cpu",
            cache_folder=str(MODEL_CACHE),
            local_files_only=True,
        )
        dense[label] = (config, model, np.load(EMBEDDING_CACHE / config["cache_file"]))

    # 4) LA QUERY LA SCRIVE ANDREA. Il ciclo continua finché scrive ``exit``.
    # Le query libere non entrano nel benchmark e non modificano i file ufficiali.
    while True:
        query = input("\nQuery finanziaria in inglese (oppure exit): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            print("La query è vuota: prova di nuovo.")
            continue

        # 5) BM25: confronto per parole con gli stessi 512 passaggi.
        bm25_results = bm25.search(query, top_k=TOP_K)
        show_ranking("BM25", [(item.passage_id, item.score) for item in bm25_results], by_id)

        # 6) ENCODER: codifichiamo soltanto la nuova query. I vettori del corpus
        # sono già nella cache. Questa è la parte eseguita a ogni domanda.
        for label, (config, model, corpus_vectors) in dense.items():
            query_vector = model.encode(
                [config["instruction"] + query],
                normalize_embeddings=True,
            )[0]

            # Ogni elemento è la similarità tra la query e un passaggio.
            scores = corpus_vectors @ query_vector
            best_rows = np.argsort(-scores, kind="stable")[:TOP_K]
            ranking = [(passages[index].passage_id, float(scores[index])) for index in best_rows]
            show_ranking(label, ranking, by_id)

        print("\nQuery esplorativa: senza qrels non calcoliamo Recall o nDCG.")


if __name__ == "__main__":
    main()
