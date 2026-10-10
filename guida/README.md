# Guida al codice

Il progetto è un sistema che, data una domanda finanziaria, trova i brani di forum che possono rispondere e fa scrivere la risposta a un modello. Questa cartella contiene versioni semplificate e commentate del codice che conta, scritte per essere lette in ordine; gli originali non sono stati toccati e la colonna di destra dice dove stanno. Ogni file gira davvero sui dati del progetto, dalla cartella `rag`, con `uv run python guida/<file>`.

## Il percorso di una domanda

```
domanda ─► BM25 (parole) ───────┐
        ├► MiniLM (significato) ├─► fusione ─► 30 candidati ─► riordino ─► 10 brani ─► prompt ─► modello ─► risposta con citazioni
        └► BGE (significato) ───┘   (RRF)                      (cross-encoder)
```

I tre metodi della prima fase scorrono tutti i 61.022 brani e sono veloci ma approssimativi; il riordinatore legge insieme domanda e brano ed è preciso ma lento, perciò vede solo i 30 candidati migliori. Il modello generativo riceve soltanto i 10 brani finali: se la fonte giusta non è fra quelli, la risposta non può essere buona.

## Ordine di lettura

| Passo | File | Che cosa si capisce | Originale |
|---|---|---|---|
| 0 | `comuni.py` | Dove stanno i dati, come si leggono, che cos'è una «posizione» in classifica | sparso in ogni script |
| 1 | `01_cerca_nel_campione.py` | Testo → vettore → somiglianza → classifica; BM25 contro BGE su 512 brani | `src/retrieval.py`, `laboratorio/d1_query.py` |
| 2 | `02_misura_la_qualita.py` | Recall, nDCG, intervallo di confidenza; misura vera di BGE su 180 domande | `esperimenti/studio-embedding-retrieval/10_baseline_180.py` |
| 3 | `ricerca.py` e `03_ricerca_completa.py` | Fusione RRF dei tre metodi e riordino; confronto con BGE da solo | `…/11_punteggi_reranker.py`, `…/16_configurazione_finale.py` |
| 4 | `04_genera_la_risposta.py` | Il prompt: brani numerati, citazioni obbligatorie, permesso di non rispondere | `esperimenti/studio-generazione/05_confronto_semplice_selezionato.py` |
| 5 | `05_agente.py` | Il ciclo pensiero → azione → osservazione, con un tetto sui passi imposto dal codice | `esperimenti/studio-agenti/01_agente_ricerca_testuale.py` |

Le chiamate al modello generativo (4 e 5) partono solo con `--chiama-api`; senza, il passo 4 stampa il prompt e il passo 5 usa un modello finto con un copione fisso, quindi si possono eseguire a costo zero.

## Il servizio (l'upgrade per ERNI)

La cartella `servizio/` è già commentata e va letta in questo ordine: `modello.py` (i sostituti simulati e gli errori), `recupero.py` (ricerca in memoria con BGE), `db.py` (stessa ricerca in PostgreSQL con pgvector), `app.py` (i due endpoint e la validazione), `__main__.py` (l'avvio). Poi `compose.yaml` e `Dockerfile` per i container, e `tests/test_servizio.py` per vedere che cosa viene verificato. Usa solo BGE su 512 brani: la fusione e il riordino della guida non sono ancora nel servizio.

## Che cosa si può ignorare

- `note/`: diario, stato e archivio delle sessioni; non descrive il codice.
- `runs/`: risultati del primo pilota su 12 domande, superato dalle misure su 180.
- `esperimenti/`: fuori dai tre script citati sopra, sono passi intermedi numerati (ogni passo è una copia del precedente con una modifica); servono solo se si vuole ricostruire come si è arrivati a una scelta.
- `laboratorio/`, `strumenti/`, `report/`: un esercizio interattivo, il banco delle decisioni (pagina HTML) e il racconto pubblico.

## Che cosa dicono i numeri

I valori stanno nel `README.md` della radice e vanno letti con le sue riserve: un solo corpus, annotazioni incomplete, ogni configurazione eseguita una volta, intervalli di circa ±0,05. In particolare il riordino porta recall@10 da 0.417 a 0.478 e la sola fusione non cambia niente (0.418), ma gli intervalli di confidenza delle due misure si sovrappongono.
