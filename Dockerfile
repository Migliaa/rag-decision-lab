# Immagine del servizio (tappa C). Contiene: Python, le dipendenze, il codice
# di servizio/ e src/, e il modello di embedding. NON contiene: dati, password,
# indirizzo del database. Quelli arrivano all'avvio del container.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Le dipendenze prima del codice: finché requirements-immagine.txt non cambia,
# Docker riusa questo strato e una modifica al codice non le reinstalla.
# L'indice aggiuntivo fornisce torch per sola CPU, molto più leggero di quello predefinito.
COPY servizio/requirements-immagine.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-immagine.txt

# Il modello viene scaricato una volta, alla revisione fissata, durante la
# costruzione: il container poi parte senza rete verso Hugging Face.
ARG MODELLO=BAAI/bge-small-en-v1.5
ARG REVISIONE=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODELLO}', revision='${REVISIONE}', device='cpu', cache_folder='/app/data/cache/huggingface')"
ENV HF_HUB_OFFLINE=1

COPY src/ src/
COPY servizio/ servizio/

# Il processo non gira come amministratore dentro il container.
RUN useradd --create-home servizio
USER servizio

EXPOSE 8010
HEALTHCHECK --interval=5s --timeout=3s --start-period=40s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=2)"

# 0.0.0.0: dentro il container bisogna ascoltare su tutte le interfacce,
# altrimenti la porta pubblicata da Docker non raggiunge il processo.
CMD ["python", "-m", "servizio", "--recupero", "pg", "--host", "0.0.0.0"]
