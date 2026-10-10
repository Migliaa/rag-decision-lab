"""Servizio HTTP della tappa A: /health e /answer; il modello è simulato.

PERCORSO DI UNA RICHIESTA A /answer
===================================

    client                      questo file                     modello.py
    ------                      -----------                     ----------
    POST /answer        ->  1. FastAPI trova la funzione
    {"question": "..."}     2. Pydantic controlla ``Domanda``
                               (se non passa: 422, e la
                               funzione non viene eseguita)
                            3. ``answer`` chiede i brani     -> recupero (simulato o recupero.py)
                            4. ``answer`` chiede la risposta -> ModelloSimulato.genera
                            5. scarta le fonti non recuperate
    200 + JSON          <-  6. Pydantic compone ``Risposta``

Se al punto 4 il modello fallisce, il client riceve 503 o 504 con un codice
d'errore stabile e il request_id; il dettaglio tecnico resta nel log.
"""

import logging
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints

from servizio.modello import (
    ModelloNonDisponibile,
    ModelloSimulato,
    ModelloTimeout,
    recupero_simulato,
)

log = logging.getLogger("servizio")

# Limite provvisorio, da decidere insieme: è il valore che Andrea modifica e
# verifica nella prima prova guidata.
LUNGHEZZA_MASSIMA_DOMANDA = 500


class Domanda(BaseModel):
    # Gli spazi ai bordi vengono tolti PRIMA del controllo di lunghezza:
    # una domanda fatta di soli spazi è quindi vuota e viene rifiutata.
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=LUNGHEZZA_MASSIMA_DOMANDA,
        ),
    ]


class Risposta(BaseModel):
    answer: str
    sources: list[str]
    request_id: str


def _errore(status: int, codice: str, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": codice, "request_id": request_id})


def crea_app(modello: ModelloSimulato | None = None, recupero=recupero_simulato) -> FastAPI:
    """Costruisce l'app attorno a un modello e a una funzione di recupero.

    I test passano qui un modello guasto o un recupero diverso senza toccare
    il resto del servizio. ``recupero`` riceve la domanda e restituisce brani.
    """
    modello = modello or ModelloSimulato()
    app = FastAPI(title="RAG FiQA — servizio locale (modello simulato)")

    @app.get("/health")
    def health() -> dict:
        # Dice solo che il processo risponde. Non interroga modello né dati.
        return {"status": "ok"}

    @app.post(
        "/answer",
        response_model=Risposta,
        responses={503: {"description": "Modello non disponibile"}, 504: {"description": "Timeout del modello"}},
    )
    def answer(domanda: Domanda):
        # Arrivati qui la domanda è già valida: non serve ricontrollarla.
        request_id = uuid4().hex
        inizio = perf_counter()

        brani = recupero(domanda.question)
        try:
            esito = modello.genera(domanda.question, brani)
        except ModelloTimeout as errore:
            log.error("request_id=%s esito=504 dettaglio=%s", request_id, errore)
            return _errore(504, "timeout_modello", request_id)
        except ModelloNonDisponibile as errore:
            log.error("request_id=%s esito=503 dettaglio=%s", request_id, errore)
            return _errore(503, "modello_non_disponibile", request_id)

        # Un modello può citare una fonte che non gli è stata data: si tengono
        # solo gli ID dei brani effettivamente recuperati.
        recuperati = {b.id for b in brani}
        fonti = [f for f in esito.fonti if f in recuperati]

        millisecondi = (perf_counter() - inizio) * 1000
        log.info("request_id=%s esito=200 fonti=%d durata_ms=%.1f", request_id, len(fonti), millisecondi)
        return Risposta(answer=esito.risposta, sources=fonti, request_id=request_id)

    return app


app = crea_app()
