# Laboratorio D1 — esplorare una query

Questo laboratorio separa ciò che Andrea deve poter cambiare dal codice di servizio.

## File da guardare, in ordine

1. `data/manifests/E001-pilot-512.json`: provenienza, hash e regola con cui è stato creato il campione.
2. `data/processed/E001-pilot-512/queries.jsonl`: le 12 query ufficiali usate nel pilot.
3. `data/processed/E001-pilot-512/corpus.jsonl`: i 512 passaggi messi a disposizione dei retriever.
4. `data/processed/E001-pilot-512/qrels.tsv`: associazioni query → passaggi pertinenti, usate solo per valutare.
5. `src/retrieval.py`: decisioni su testo indicizzato, BM25, istruzione BGE, normalizzazione e ranking.
6. `laboratorio/d1_query.py`: punto di ingresso per provare una domanda scelta da Andrea.

I file originali scaricati vivono sotto `data/raw/` e non vengono modificati. Il campione in
`data/processed/` viene rigenerato dallo script `scripts/prepare_fiqa_pilot.py`.

## Provare una query

Avviare `uv run python -m laboratorio.d1_query`. Il programma accetta più frasi, le cerca con
BM25, MiniLM e BGE e stampa i primi cinque passaggi; `exit` termina il laboratorio. Le query
libere non possiedono una risposta gold: servono a esplorare il comportamento e non entrano
nelle metriche ufficiali.
