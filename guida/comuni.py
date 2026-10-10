"""Parti condivise dagli esempi della guida: dove stanno i file e come si leggono.

Qui non c'è nessuna logica di ricerca, solo percorsi e piccole funzioni che
leggono i dati. Gli esempi numerati la importano con: from comuni import ...
"""

import json
import os
import re
from pathlib import Path

import numpy as np

# I modelli sono già scaricati in data/cache. Questa riga impedisce a Python di
# collegarsi a internet per controllare se esistono versioni nuove: se un
# modello mancasse, l'errore sarebbe subito chiaro invece di una lunga attesa.
# Va eseguita prima di importare sentence_transformers, per questo sta qui.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ---------------------------------------------------------------------------
# DOVE STANNO I FILE
# ---------------------------------------------------------------------------
RADICE = Path(__file__).resolve().parents[1]  # la cartella "rag"
CACHE_MODELLI = RADICE / "data" / "cache" / "huggingface"

# Il corpus completo: 61.022 brani di forum finanziari (FiQA).
CORPUS_COMPLETO = RADICE / "data" / "raw" / "fiqa" / "fiqa.jsonl"

# Le domande e le "fonti giuste" (annotate a mano da chi ha costruito il dataset).
DOMANDE_DIR = RADICE / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
FONTI_GIUSTE = DOMANDE_DIR / "qrels" / "dev.tsv"

# I vettori dei 61.022 brani, calcolati una volta sola dagli esperimenti.
VETTORI_DIR = RADICE / "esperimenti" / "studio-embedding-retrieval" / "output"

# Un campione piccolo (512 brani) con i suoi vettori: serve per provare in fretta.
CAMPIONE_DIR = RADICE / "data" / "processed" / "E001-pilot-512"
VETTORI_CAMPIONE_DIR = RADICE / "data" / "cache" / "d1-lab-512"

# Il file con la chiave del modello generativo (una riga: GEMINI_API=...). Non è in Git.
FILE_CHIAVE = RADICE / "esperimenti" / "studio-generazione" / "apikey.env"

# Prima di confrontare una domanda con i brani, BGE vuole questa frase davanti.
# Serve solo per le domande: i brani si codificano senza.
PREFISSO_BGE = "Represent this sentence for searching relevant passages: "

# I due modelli che trasformano un testo in 384 numeri. "revisione" fissa la
# versione esatta del modello, così nessun aggiornamento cambia i risultati.
MODELLI = {
    "minilm": {
        "nome": "sentence-transformers/all-MiniLM-L6-v2",
        "revisione": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "prefisso_domanda": "",
        "file_vettori": "corpus_embeddings_minilm_full.npy",
    },
    "bge": {
        "nome": "BAAI/bge-small-en-v1.5",
        "revisione": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "prefisso_domanda": PREFISSO_BGE,
        "file_vettori": "corpus_embeddings_bge_full.npy",
    },
}


# ---------------------------------------------------------------------------
# LETTURA DEI DATI
# ---------------------------------------------------------------------------
def leggi_jsonl(percorso):
    """Un file .jsonl ha un oggetto JSON per riga. Restituisce la lista degli oggetti."""
    righe = []
    with open(percorso, encoding="utf-8") as file:
        for riga in file:
            if riga.strip():
                righe.append(json.loads(riga))
    return righe


def testo_del_brano(brano):
    """Il testo che i modelli leggono: il titolo (se c'è) seguito dal testo."""
    if brano.get("title"):
        return brano["title"] + "\n" + brano["text"]
    return brano["text"]


def carica_corpus(percorso=CORPUS_COMPLETO):
    """Restituisce due liste parallele: gli id dei brani e i loro testi."""
    ids = []
    testi = []
    for brano in leggi_jsonl(percorso):
        ids.append(brano["_id"])
        testi.append(testo_del_brano(brano))
    return ids, testi


def carica_domande(nome_file):
    """Legge un file di domande e restituisce {id_domanda: testo}.

    In MTRAG ogni domanda esiste in più forme, tutte nella stessa cartella:
    - fiqa_rewrite: la domanda riscritta per essere comprensibile da sola
    - fiqa_lastturn: solo l'ultima frase dell'utente, senza il contesto
    """
    domande = {}
    for riga in leggi_jsonl(DOMANDE_DIR / f"{nome_file}.jsonl"):
        domande[riga["_id"]] = riga["text"].replace("|user|:", " ").strip()
    return domande


def carica_fonti_giuste():
    """Restituisce {id_domanda: insieme di id dei brani che rispondono}."""
    fonti = {}
    with open(FONTI_GIUSTE, encoding="utf-8") as file:
        next(file)  # la prima riga contiene i nomi delle colonne
        for riga in file:
            id_domanda, id_brano, punteggio = riga.rstrip("\n").split("\t")
            if int(punteggio) > 0:
                fonti.setdefault(id_domanda, set()).add(id_brano)
    return fonti


# ---------------------------------------------------------------------------
# DUE FUNZIONI USATE OVUNQUE
# ---------------------------------------------------------------------------
def parole(testo):
    """Spezza un testo in parole minuscole. È la vista del testo che ha BM25.

    Il modello a pattern tiene insieme sigle e numeri come "zx-401" o "1.5".
    """
    return re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", testo.lower())


def posizioni(punteggi):
    """Per ogni brano dice in che posizione è arrivato (1 = il migliore).

    Esempio: punteggi [0.2, 0.9, 0.5] -> posizioni [3, 1, 2].
    Tutta la valutazione e la fusione fra metodi lavorano su queste posizioni,
    non sui punteggi, perché i punteggi di metodi diversi non sono confrontabili.
    """
    ordine = np.argsort(-punteggi, kind="stable")  # indici dal migliore al peggiore
    risultato = np.empty(len(ordine), dtype=np.int64)
    risultato[ordine] = np.arange(1, len(ordine) + 1)
    return risultato


def leggi_chiave():
    """Legge GEMINI_API dal file apikey.env. Serve solo se si chiama il modello generativo."""
    for riga in FILE_CHIAVE.read_text(encoding="utf-8").splitlines():
        nome, _, valore = riga.partition("=")
        if nome.strip() == "GEMINI_API":
            return valore.strip()
    raise RuntimeError(f"GEMINI_API non trovata in {FILE_CHIAVE}")
