"""Step 1b: indicizzare lo stesso corpus con un modello diverso (BGE-small).

Identico a 01_indicizzazione.py (stesso corpus, stessa logica), cambia solo
il modello di embedding. Serve per rispondere a una domanda diversa da
quella dello step 1: non "come si indicizza", ma "due modelli diversi
mettono gli stessi testi in punti diversi dello spazio?" — la matrice qui
prodotta è confrontabile con quella MiniLM solo dopo una riduzione a 2D/3D
(vedi 02_visualizza_embedding.py), MAI sommando o confrontando direttamente
i numeri grezzi: i due modelli non condividono lo stesso spazio a 384
dimensioni, condividono solo il numero di dimensioni per coincidenza.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/01b_indicizzazione_bge.py
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "E001-pilot-512" / "corpus.jsonl"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # fissata: non cambia da sola

# BGE distingue un ruolo "query" da un ruolo "passaggio": la model card
# raccomanda un prefisso istruzione SOLO quando si incorpora una domanda,
# mai quando si incorpora un passaggio da indicizzare (qui). Lo teniamo
# esplicito ora perché conterà allo step delle query, non a questo.
PASSAGE_PREFIX = ""  # nessun prefisso per i passaggi del corpus


def load_corpus(path: Path) -> list[dict]:
    passages = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages


def indexed_text(passage: dict) -> str:
    title = passage.get("title", "")
    text = passage["text"]
    return f"{title}\n{text}" if title else text


def main() -> None:
    passages = load_corpus(CORPUS_PATH)
    print(f"Passaggi caricati: {len(passages)}")

    print("\nCarico BGE-small (da cache locale, nessun download)...")
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
        cache_folder=str(CACHE_HF),
    )

    texts = [PASSAGE_PREFIX + indexed_text(p) for p in passages]

    print(f"\nIncorporo {len(texts)} passaggi (batch da 16, una tantum)...")
    started = perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = perf_counter() - started

    print(f"\nFatto in {elapsed:.1f} secondi.")
    print(f"Forma della matrice risultante: {embeddings.shape}")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Controllo: norma media dei vettori = {norms.mean():.4f} (atteso: 1.0)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "corpus_embeddings_bge.npy"
    np.save(out_path, embeddings)
    print(f"\nMatrice salvata in: {out_path}")

    # Stessi id, stesso ordine di 01_indicizzazione.py: il corpus è lo
    # stesso file, letto nello stesso modo. Salvarli di nuovo qui (invece di
    # assumere che il file di 01 esista ancora) rende questo script eseguibile
    # da solo, senza dipendere dall'ordine di esecuzione degli altri.
    ids_path = OUT_DIR / "corpus_ids.json"
    ids_path.write_text(json.dumps([p["_id"] for p in passages]), encoding="utf-8")
    print(f"Id salvati in: {ids_path}")


if __name__ == "__main__":
    main()
