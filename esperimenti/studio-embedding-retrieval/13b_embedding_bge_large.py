"""Step 13b: incorporare il corpus con bge-large-en-v1.5.

Sostituisce il tentativo fallito con gte-base-en-v1.5 (vedi 13_embedding_modello_forte.py):
quel modello e' piu' forte sulla carta ma richiede codice proprio, incompatibile con la
versione di transformers installata. bge-large usa un'architettura BERT standard, e' della
stessa famiglia di bge-small gia' in uso - stesso prefisso per le domande, stessa
convenzione - e su BEIR/FiQA e' dato intorno a 0.45 di nDCG@10 contro 0.403 di bge-small.

Costo misurato: circa 1,3 passaggi al secondo su testi lunghi, cioe' fra le sei e le tredici
ore per i 61.022 passaggi a seconda della lunghezza reale. Salva un checkpoint ogni mille
passaggi e riprende da solo: il portatile si e' gia' spento piu' volte durante run simili.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/13b_embedding_bge_large.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

MODEL_KEY = "bge_large"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
MAX_SEQ_LENGTH = 512
CHUNK_SIZE = 1000
BATCH_SIZE = 16


def indexed_text(passage: dict) -> str:
    title = passage.get("title", "")
    text = passage["text"]
    return f"{title}\n{text}" if title else text


def load_corpus(path: Path) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            texts.append(indexed_text(json.loads(line)))
    return texts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    texts = load_corpus(CORPUS_PATH)
    total = len(texts)

    emb_path = OUT_DIR / f"corpus_embeddings_{MODEL_KEY}_full.npy"
    progress_path = OUT_DIR / f"{MODEL_KEY}_full_progress.json"

    print(f"Carico {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cpu", cache_folder=str(CACHE_HF))
    model.max_seq_length = MAX_SEQ_LENGTH
    dim = model.get_sentence_embedding_dimension()
    print(f"Passaggi: {total}   dimensione: {dim}   finestra: {model.max_seq_length}")

    if emb_path.exists() and progress_path.exists():
        embeddings = np.load(emb_path)
        start = json.loads(progress_path.read_text())["completed"]
        print(f"Riprendo da {start}/{total}")
    else:
        embeddings = np.zeros((total, dim), dtype=np.float32)
        start = 0

    t0 = time.time()
    for begin in range(start, total, CHUNK_SIZE):
        end = min(begin + CHUNK_SIZE, total)
        embeddings[begin:end] = model.encode(texts[begin:end], normalize_embeddings=True,
                                             batch_size=BATCH_SIZE, show_progress_bar=False)
        np.save(emb_path, embeddings)
        progress_path.write_text(json.dumps({"completed": end, "total": total, "model": MODEL_NAME}))
        speed = (end - start) / (time.time() - t0)
        print(f"  {end}/{total}  ({speed:.1f} passaggi/s, ~{(total - end) / speed / 3600:.1f} ore rimanenti)",
              flush=True)

    print(f"Completato: {emb_path}")


if __name__ == "__main__":
    main()
