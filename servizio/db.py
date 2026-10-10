"""Persistenza della tappa B: brani e vettori in PostgreSQL con pgvector.

Una sola tabella, ``brani``: una riga per brano, con il testo, il vettore e la
provenienza (campione, modello e revisione che hanno prodotto il vettore).

    importazione (una volta)   corpus.jsonl + bge.npy  ->  INSERT nella tabella
    a ogni domanda             domanda -> modello -> vettore [384]
                               SELECT ... ORDER BY embedding <=> vettore LIMIT k

``<=>`` è l'operatore di pgvector per la distanza coseno: 0 se due vettori
puntano nella stessa direzione, valori più alti se divergono. Ordinare per
distanza crescente equivale a ordinare per similarità decrescente, cioè alla
classifica che ``recupero.py`` calcola in memoria.

Nessun indice approssimato: con 512 righe PostgreSQL confronta la domanda con
tutti i vettori (ricerca esatta), quindi il risultato è lo stesso del calcolo
in memoria.
"""

import os
from pathlib import Path

import psycopg

from servizio.modello import Brano

FILE_ENV = Path(__file__).resolve().parents[1] / ".env"

DIMENSIONI = 384

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS brani (
    id         text PRIMARY KEY,
    testo      text NOT NULL,
    embedding  vector({DIMENSIONI}) NOT NULL,
    campione   text NOT NULL,
    modello    text NOT NULL,
    revisione  text NOT NULL
);
"""

# La chiave primaria su ``id`` rende l'importazione ripetibile: se il brano
# esiste già la riga viene aggiornata, non duplicata.
INSERISCI = """
INSERT INTO brani (id, testo, embedding, campione, modello, revisione)
VALUES (%s, %s, %s::vector, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    testo = EXCLUDED.testo,
    embedding = EXCLUDED.embedding,
    campione = EXCLUDED.campione,
    modello = EXCLUDED.modello,
    revisione = EXCLUDED.revisione
"""

CERCA = "SELECT id, testo FROM brani ORDER BY embedding <=> %s::vector, id LIMIT %s"


def url_database() -> str:
    """Indirizzo del database, mai scritto nel codice.

    Nel container arriva già pronto in ``DATABASE_URL``. Fuori da Docker si
    costruisce l'indirizzo del database locale di compose.yaml leggendo la
    password da ``.env``, lo stesso file che usa Compose.
    """
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]
    password = os.environ.get("POSTGRES_PASSWORD")
    if password is None and FILE_ENV.exists():
        for riga in FILE_ENV.read_text(encoding="utf-8").splitlines():
            nome, _, valore = riga.partition("=")
            if nome.strip() == "POSTGRES_PASSWORD":
                password = valore.strip()
    if not password:
        raise RuntimeError("password del database assente: creare .env a partire da .env.example")
    return f"postgresql://rag:{password}@127.0.0.1:5433/rag"


def connetti() -> psycopg.Connection:
    return psycopg.connect(url_database(), connect_timeout=5)


def come_vettore(valori) -> str:
    # pgvector accetta un vettore scritto come testo: '[0.1,0.2,...]'.
    return "[" + ",".join(f"{float(v):.8f}" for v in valori) + "]"


def importa(connessione: psycopg.Connection) -> int:
    """Crea la tabella se manca e carica brani e vettori del campione. Restituisce le righe presenti."""
    # Import qui: i percorsi e i nomi del modello vivono in recupero.py, ma
    # l'importazione non deve caricare il modello in memoria.
    import json

    import numpy as np

    from servizio.recupero import CORPUS, MODELLO, REVISIONE, VETTORI
    from src.retrieval import Passage

    with CORPUS.open(encoding="utf-8") as righe:
        record = [json.loads(riga) for riga in righe if riga.strip()]
    vettori = np.load(VETTORI)
    if len(record) != vettori.shape[0] or vettori.shape[1] != DIMENSIONI:
        raise ValueError(f"{len(record)} brani e vettori {vettori.shape}: file non allineati")

    campione = CORPUS.parent.name
    with connessione.cursor() as cursore:
        cursore.execute(SCHEMA)
        cursore.executemany(
            INSERISCI,
            [
                (r["_id"], Passage(r["_id"], r.get("title", ""), r["text"]).indexed_text,
                 come_vettore(v), campione, MODELLO, REVISIONE)
                for r, v in zip(record, vettori)
            ],
        )
        cursore.execute("SELECT count(*) FROM brani")
        presenti = cursore.fetchone()[0]
    connessione.commit()
    return presenti


class RecuperoPg:
    """Stessa forma di ``RecuperoBge``: riceve la domanda, restituisce i brani."""

    def __init__(self, k: int | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        from servizio.recupero import CACHE_MODELLI, K, MODELLO, REVISIONE

        self.k = k or K
        self.modello = SentenceTransformer(
            MODELLO, revision=REVISIONE, device="cpu", cache_folder=str(CACHE_MODELLI), local_files_only=True
        )
        # Fallisce subito all'avvio, con un messaggio chiaro, se il database è spento o vuoto.
        with connetti() as connessione:
            presenti = connessione.execute("SELECT count(*) FROM brani").fetchone()[0]
        if presenti == 0:
            raise RuntimeError("tabella brani vuota: eseguire python -m servizio.db")

    def __call__(self, domanda: str) -> list[Brano]:
        from src.retrieval import BGE_QUERY_INSTRUCTION

        vettore = self.modello.encode([BGE_QUERY_INSTRUCTION + domanda], normalize_embeddings=True)[0]
        # Una connessione per richiesta: semplice e sufficiente in locale.
        with connetti() as connessione:
            righe = connessione.execute(CERCA, (come_vettore(vettore), self.k)).fetchall()
        return [Brano(id, testo) for id, testo in righe]


if __name__ == "__main__":
    with connetti() as connessione:
        print(f"brani nel database: {importa(connessione)}")
