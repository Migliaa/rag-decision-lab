"""Motore di ricerca sperimentale usato dall'infrastruttura D1.

Questo NON è un RAG completo: non contiene un modello generativo e non produce
risposte. Riceve una domanda e restituisce una classifica di passaggi.

MAPPA DEL FILE
==============

1. ``Passage`` e ``RankedPassage`` descrivono input e output.
2. ``BM25Retriever`` è il ramo lessicale: confronta parole e frequenze.
3. ``DenseRetriever`` è il ramo embedding:

   preparazione una tantum
       testi dei passaggi -> encoder -> matrice [passaggi, 384]

   per ogni nuova domanda
       query -> stesso encoder -> vettore [384]
       matrice @ vettore -> un punteggio per passaggio -> top-k

Il caricamento del dataset, le qrels, le metriche e le figure vivono in altri
file. Questa separazione è utile nel software, ma non è un buon ordine per una
prima lezione: il laboratorio ripartirà prima dai singoli dati e vettori.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# BGE è stato addestrato a distinguere due ruoli:
# - la query breve, che chiede di cercare;
# - il passaggio, che è una possibile fonte.
# La model card raccomanda questo prefisso SOLO per la query.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class Passage:
    passage_id: str
    title: str
    text: str

    @property
    def indexed_text(self) -> str:
        """Decide quale testo rappresentare.

        In D1 non riscriviamo e non riassumiamo le fonti: usiamo il passaggio
        ufficiale, preceduto dal titolo quando il dataset ne fornisce uno.
        Cambiare questa funzione cambierebbe ciò che BM25 e gli encoder vedono.
        """
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass(frozen=True)
class RankedPassage:
    passage_id: str
    score: float


def lexical_tokens(text: str) -> list[str]:
    """Divide il testo nelle unità che BM25 può contare.

    Il pattern conserva numeri decimali e identificatori come ``ZX-401``.
    BM25 non riceve il testo originale: riceve questa lista di token.
    """
    return re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower())


class BM25Retriever:
    def __init__(self, passages: Sequence[Passage]) -> None:
        self.passages = list(passages)
        # L'indice conta in quali documenti compaiono i token e quanto sono rari.
        # Questa è preparazione una tantum: non viene rifatta per ogni domanda.
        self.index = BM25Okapi([lexical_tokens(p.indexed_text) for p in passages])

    def search(self, query: str, top_k: int = 10) -> list[RankedPassage]:
        # La query attraversa la STESSA tokenizzazione dei documenti.
        scores = np.asarray(self.index.get_scores(lexical_tokens(query)))
        # argsort(-scores) mette prima i punteggi più alti. ``stable`` rende
        # riproducibile l'ordine quando due documenti hanno lo stesso punteggio.
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [RankedPassage(self.passages[i].passage_id, float(scores[i])) for i in order]


class DenseRetriever:
    def __init__(
        self,
        passages: Sequence[Passage],
        model_name: str,
        revision: str,
        query_instruction: str = "",
        batch_size: int = 16,
        cache_folder: str | None = None,
    ) -> None:
        self.passages = list(passages)
        # Carichiamo una revisione precisa del modello, sempre su CPU.
        # ``revision`` impedisce che un futuro aggiornamento silenzioso del
        # checkpoint cambi l'esperimento.
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device="cpu",
            cache_folder=cache_folder,
        )
        self.query_instruction = query_instruction
        self.batch_size = batch_size

        started = perf_counter()
        # Ogni passaggio diventa un vettore di 384 numeri. La normalizzazione
        # porta ogni vettore a lunghezza 1 e ci permette di usare il prodotto
        # scalare come similarità coseno.
        self.embeddings = self.model.encode(
            [p.indexed_text for p in passages],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.encoding_seconds = perf_counter() - started

    def search(self, query: str, top_k: int = 10) -> list[RankedPassage]:
        # Solo BGE riceve un prefisso; per MiniLM ``query_instruction`` è vuoto.
        query_vector = self.model.encode(
            [self.query_instruction + query],
            normalize_embeddings=True,
        )[0]
        # Una moltiplicazione confronta la query con TUTTI i passaggi.
        # Il calcolo avviene nei 384 valori originali, non nella futura mappa 2D.
        scores = self.embeddings @ query_vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [RankedPassage(self.passages[i].passage_id, float(scores[i])) for i in order]
