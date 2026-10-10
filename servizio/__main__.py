"""Avvio locale: ``python -m servizio [--recupero pg|bge|simulato] [--guasto indisponibile|timeout] [--porta N]``.

Uvicorn è il server che ascolta sulla porta e passa ogni richiesta all'app
FastAPI. Per default ascolta su 127.0.0.1: il servizio è raggiungibile solo
da questo computer. Nel container si usa ``--host 0.0.0.0``.
"""

import argparse
import logging
import os

import uvicorn

from servizio.app import crea_app
from servizio.modello import GUASTI, ModelloSimulato, recupero_simulato

parser = argparse.ArgumentParser(description="Servizio RAG locale con modello simulato")
parser.add_argument(
    "--guasto",
    choices=GUASTI,
    default=os.environ.get("GUASTO_MODELLO") or None,
    help="fa fallire il modello simulato a ogni richiesta; si può impostare anche con la variabile GUASTO_MODELLO",
)
parser.add_argument(
    "--recupero",
    choices=("pg", "bge", "simulato"),
    default=os.environ.get("RECUPERO_MODALITA", "bge"),
    help="(variabile RECUPERO_MODALITA) pg: ricerca in PostgreSQL/pgvector; bge: stessi vettori tenuti in memoria; simulato: due brani fissi inventati",
)
parser.add_argument("--porta", type=int, default=8010)
parser.add_argument("--host", default="127.0.0.1")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
if args.recupero == "bge":
    # Import qui: caricare il modello richiede qualche secondo e non serve al recupero simulato.
    from servizio.recupero import RecuperoBge

    recupero = RecuperoBge()
elif args.recupero == "pg":
    from servizio.db import RecuperoPg

    recupero = RecuperoPg()
else:
    recupero = recupero_simulato

uvicorn.run(crea_app(ModelloSimulato(guasto=args.guasto), recupero), host=args.host, port=args.porta)
