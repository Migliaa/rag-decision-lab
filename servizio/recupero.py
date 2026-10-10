"""Recupero reale per il servizio: ``bge-small-en-v1.5`` sul campione da 512 brani.

Riusa ciò che gli esperimenti hanno già prodotto, senza ricalcolare nulla:

- i brani di ``data/processed/E001-pilot-512/corpus.jsonl``;
- i loro vettori in ``data/cache/d1-lab-512/bge.npy`` (una riga per brano,
  nello stesso ordine del file);
- il modello già scaricato in ``data/cache/huggingface``, caricato senza rete.

    all'avvio (una volta)      brani + matrice [512, 384] + modello in memoria
    a ogni domanda             domanda -> modello -> vettore [384]
                               matrice @ vettore -> 512 punteggi -> primi k

Le misure del README (recall@10 0.417) riguardano i 61.022 brani: su questo
campione non valgono.
"""

import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from servizio.modello import Brano
from src.retrieval import BGE_QUERY_INSTRUCTION, Passage

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/E001-pilot-512/corpus.jsonl"
VETTORI = ROOT / "data/cache/d1-lab-512/bge.npy"
CACHE_MODELLI = ROOT / "data/cache/huggingface"

# Nome e revisione devono essere quelli con cui sono stati calcolati i vettori
# salvati: una domanda codificata da un altro modello non sarebbe confrontabile.
MODELLO = "BAAI/bge-small-en-v1.5"
REVISIONE = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

# Quanti brani passare al modello. Si cambia dall'esterno con la variabile
# d'ambiente RECUPERO_K, senza toccare il codice né ricostruire l'immagine.
K = int(os.environ.get("RECUPERO_K", "5"))
if K < 1:
    raise ValueError(f"RECUPERO_K deve essere almeno 1, ricevuto {K}")


class RecuperoBge:
    def __init__(self) -> None:
        with CORPUS.open(encoding="utf-8") as righe:
            record = [json.loads(riga) for riga in righe if riga.strip()]
        # ``indexed_text`` è lo stesso testo (titolo + brano) codificato negli esperimenti.
        self.brani = [Brano(r["_id"], Passage(r["_id"], r.get("title", ""), r["text"]).indexed_text) for r in record]
        self.vettori = np.load(VETTORI)
        if len(self.brani) != self.vettori.shape[0]:
            raise ValueError(f"{len(self.brani)} brani ma {self.vettori.shape[0]} vettori: file non allineati")
        self.modello = SentenceTransformer(
            MODELLO,
            revision=REVISIONE,
            device="cpu",
            cache_folder=str(CACHE_MODELLI),
            local_files_only=True,
        )

    def __call__(self, domanda: str) -> list[Brano]:
        vettore = self.modello.encode([BGE_QUERY_INSTRUCTION + domanda], normalize_embeddings=True)[0]
        punteggi = self.vettori @ vettore
        migliori = np.argsort(-punteggi, kind="stable")[:K]
        return [self.brani[i] for i in migliori]
