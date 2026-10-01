"""Step 13: incorporare il corpus con un modello di embedding piu' forte, sempre gratuito.

NOTA SULL'AMBIENTE, importante per riprodurre questo passo. Con `transformers` 5.17 questo
script falliva: il modello richiede `trust_remote_code=True`, cioe' porta con se' il proprio
codice invece di usare un'architettura standard della libreria, e quel codice si rompeva due
volte - un buffer di indici non persistente materializzato da memoria non inizializzata
(`IndexError: index 2738746490880 is out of bounds`), e una chiamata a
`get_extended_attention_mask`, metodo rimosso nella serie 5.

Il problema si e' risolto da solo installando `optimum[onnxruntime]`, che ha riportato
`transformers` alla 4.57.6: la versione per cui quel codice era stato scritto. Verificato che
il downgrade non alteri nulla del lavoro gia' fatto (gli embedding di bge-small ricalcolati
coincidono con la cache precedente, similarita' coseno 1.000000).

Resta la lezione: `trust_remote_code` non e' solo una questione di fiducia nell'autore, che e'
il modo in cui viene di solito presentata. E' una dipendenza da codice che invecchia
separatamente dalla libreria che lo ospita, e che quindi vincola la versione della libreria -
un modello con architettura standard resta utilizzabile per anni, uno con codice proprio no.

Velocita' misurata su questa CPU: 2,65 passaggi/s, circa 6,4 ore per i 61.022 passaggi -
contro le 23 ore di bge-large-en-v1.5 (vedi 13b), che ha piu' del doppio dei parametri ed e'
anche piu' debole su FiQA.

Motivo: sulle 180 domande BGE-small (0.417 di recall@10) batte MiniLM (0.369), coerente
con i benchmark pubblici, e il divario fra i modelli e' il piu' grande osservato finora
fra tutte le leve provate. Sul benchmark BEIR/FiQA, gte-base-en-v1.5 riporta nDCG@10 di
0.487 contro 0.403 di bge-small-en-v1.5: se una frazione di quel divario si trasferisce
qui, e' il guadagno singolo piu' grande disponibile a costo zero in licenza.

Costo reale: il modello ha 137M parametri contro 33M, quindi incorporare i 61.022
passaggi richiede diverse ore di CPU. Lo script salva un checkpoint dopo ogni blocco e
riprende da solo se il processo viene interrotto (il portatile si e' gia' spento piu'
volte durante run simili).

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/13_embedding_modello_forte.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

MODEL_KEY = "gte-base"
MODEL_NAME = "Alibaba-NLP/gte-base-en-v1.5"
QUERY_PREFIX = ""          # gte-v1.5 non usa istruzioni ne' prefissi
MAX_SEQ_LENGTH = 512       # il modello arriva a 8192, ma i passaggi qui stanno sotto 512
CHUNK_SIZE = 1000
TRUST_REMOTE_CODE = True   # richiesto da gte-v1.5: scarica ed esegue codice del modello da HF


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


def acquire_lock() -> bool:
    """Evita che due esecuzioni scrivano sullo stesso checkpoint.

    Serve davvero: dopo una serie di riavvii per spegnimenti del portatile si sono
    trovati quattro processi identici che calcolavano insieme e salvavano a turno lo
    stesso array. Qui i dati sono rimasti integri per fortuna - ogni processo riscrive
    l'intero array e l'ultimo a salvare era il piu' avanti - ma il lavoro duplicato ha
    sprecato ore di CPU e la corruzione era possibile.
    """
    import psutil
    lock = OUT_DIR / f"{MODEL_KEY}_full.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                if any("13_embedding" in str(a) for a in (proc.cmdline() or [])):
                    print(f"Gia' in esecuzione (pid {pid}). Esco senza toccare il checkpoint.")
                    return False
        except Exception:
            pass
    lock.write_text(str(os.getpid()))
    return True


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not acquire_lock():
        return
    ids, texts = load_corpus(CORPUS_PATH)
    total = len(texts)
    print(f"Passaggi: {total}")

    emb_path = OUT_DIR / f"corpus_embeddings_{MODEL_KEY}_full.npy"
    progress_path = OUT_DIR / f"{MODEL_KEY}_full_progress.json"

    print(f"Carico {MODEL_NAME} (primo avvio: scarica i pesi)...")
    model = SentenceTransformer(MODEL_NAME, device="cpu", cache_folder=str(CACHE_HF),
                                trust_remote_code=TRUST_REMOTE_CODE)
    model.max_seq_length = MAX_SEQ_LENGTH
    dim = model.get_sentence_embedding_dimension()
    print(f"Dimensione degli embedding: {dim}   finestra: {model.max_seq_length} token")

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
                                             batch_size=16, show_progress_bar=False)
        np.save(emb_path, embeddings)
        progress_path.write_text(json.dumps({"completed": end, "total": total, "model": MODEL_NAME}))
        done = end - start
        speed = done / (time.time() - t0)
        print(f"  {end}/{total}  ({speed:.1f} passaggi/s, ~{(total - end) / speed / 60:.0f} min rimanenti)")

    print(f"Completato: {emb_path}")


if __name__ == "__main__":
    main()
