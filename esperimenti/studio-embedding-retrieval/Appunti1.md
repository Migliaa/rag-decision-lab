# Appunti 1 — Scelta del modello di embedding

Figura di riferimento: **Fig1.1** = `output/embedding_2d_tsne_categorie.png` (script
`02b_visualizza_embedding_tsne.py`). Indice di tutti gli Appunti: `INDICE_APPUNTI.md`.

Nota di lavoro dello studio libero, da confluire nel report del modulo quando la consolidazione
dei risultati lo richiederà (per convenzione del progetto, `docs/DOCUMENTAZIONE.md`, il report per
modulo nasce quando esistono risultati verificati, non durante l'esplorazione).

## Le due tecnologie confrontate nell'esercizio (Fig1.1)

**MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) — 22 milioni di parametri, 384 dimensioni
per vettore, addestramento generico non specializzato su un dominio. **BGE-small**
(`BAAI/bge-small-en-v1.5`) — 33 milioni di parametri, stessa dimensione del vettore per
coincidenza (non per vincolo tra i due modelli), addestrato con una tecnica contrastiva pensata
apposta per il retrieval, con un prefisso istruzione dedicato per le query (non per i passaggi:
vedi `01b_indicizzazione_bge.py`).

Misurato su questi 512 passaggi, stessa CPU, stesso batch: 16.5 secondi per indicizzare con
MiniLM, 39.1 secondi con BGE — quasi 2.4 volte più lento, coerente con l'avere il 50% di parametri
in più. È un costo, non una misura di qualità.

Fig1.1 mostra i 31 passaggi su 512 con un argomento noto dalle qrels del pilota, colorati per
conversazione di origine; il resto (grigio) sono passaggi senza argomento noto in questo
sottoinsieme, non passaggi "senza argomento". In entrambi i modelli i passaggi della stessa
conversazione tendono a restare vicini tra loro. La proiezione t-SNE distorce le distanze globali
per costruzione ed è calcolata con un solo seed — una verifica di sensibilità con un secondo
seed/perplexity non è ancora stata fatta. **Limite scoperto dopo, in Appunti2:** i colori qui
raggruppano per intera conversazione, non per singolo turno — due turni della stessa conversazione
possono avere oro diverso (visto lavorando sulla domanda "What is an IRA?", turno 3 di una
conversazione il cui turno 2 riguardava la tassazione del 401k). Per misure precise su una singola
domanda si usa sempre l'oro esatto di quel turno, non quello dell'intera conversazione.

## Le alternative reali del mercato — perché contano anche quando non le usiamo qui

Questa sezione esiste apposta per il mondo del lavoro, non solo per l'esercizio: un'azienda che
costruisce un RAG raramente parte da MiniLM. La scelta reale oscilla quasi sempre tra un modello
open-weights scaricabile (quello che facciamo qui) e un'**API di embedding a pagamento** — ed è
una decisione di design vera, con argomenti concreti da entrambe le parti, non un'opzione scartata
a priori.

### Modelli open-weights più grandi della stessa famiglia

| Modello | Dimensioni | Nota |
|---|---|---|
| `bge-base-en-v1.5` / `bge-large-en-v1.5` | 768 / 1024 | Stesso addestramento di BGE-small, punteggio più alto su MTEB (il benchmark pubblico di riferimento per il retrieval), più tempo di indicizzazione/query |
| E5 (`intfloat/e5-base-v2`, `e5-large-v2`) | 768 / 1024 | Microsoft, stesso schema a prefisso istruzione di BGE ("query: " / "passage: ") |
| GTE (`thenlp/gte-base`, `gte-large`) | 768 / 1024 | Alibaba, nessun prefisso richiesto, risultati comparabili a parità di dimensione |
| `nomic-embed-text-v1.5` | 768 | Contesto fino a 8192 token (contro i 512 di MiniLM/BGE) — utile se i passaggi non fossero già pre-tagliati come i nostri |

### API a pagamento — la scelta più diffusa in produzione

Non sono un'alternativa marginale: nella pratica professionale sono spesso la prima scelta,
perché un team non deve gestire GPU/CPU per servire il modello, ottiene aggiornamenti del modello
senza dover ripetere il fine-tuning, e i modelli di punta di questi provider occupano posizioni
alte nella classifica MTEB.

- **OpenAI `text-embedding-3-small` / `text-embedding-3-large`** — 1536 / 3072 dimensioni di
  default, con supporto nativo a troncare il vettore a una dimensione più piccola dichiarata in
  fase di richiesta (tecnica "Matryoshka": il modello è addestrato in modo che i primi N numeri del
  vettore restino un embedding valido da soli) mantenendo buona parte della qualità — un
  compromesso costo/qualità scelto a runtime, non in fase di addestramento. Prezzo per milione di
  token processati, non per richiesta.
- **Cohere `embed-v3`** — famiglie multilingue dedicate, e un parametro esplicito `input_type`
  (`search_query` vs `search_document`) che nell'API sostituisce quello che in BGE è un prefisso
  di testo scritto a mano: stessa idea (query e documento non sono trattati allo stesso modo),
  automatizzata dal provider.
- **Voyage AI** — modelli generici (`voyage-3`) e **modelli di dominio**: `voyage-finance-2`
  (addestrato specificamente su testi finanziari, il dominio di questo stesso progetto),
  `voyage-law-2` per il legale. Spesso citato insieme a Claude/Anthropic negli stack RAG perché
  entrambi ottimizzati per lavorare insieme.
- **Google Vertex AI (`text-embedding-004`), Azure OpenAI embeddings** — stessa categoria, scelta
  spesso per restare nello stesso fornitore cloud già in uso per il resto dell'infrastruttura.

### Il criterio di scelta reale (non "il più in alto in classifica")

| Fattore | Favorisce open-weights locale | Favorisce API a pagamento |
|---|---|---|
| Privacy/compliance | Dati mai usciti dall'infrastruttura — spesso un vincolo non negoziabile in finance/sanità | — |
| Volume di richieste | Costo fisso (hardware) conviene a volumi altissimi | Nessun costo di infrastruttura, conviene a volumi bassi/medi o carichi imprevedibili |
| Qualità di punta | Serve fine-tuning o un modello locale molto grande per avvicinarsi | Spesso già ai vertici MTEB senza sforzo di training |
| Manutenzione | Un team deve gestire versioni, GPU, scaling | Il provider gestisce tutto, ma introduce un fornitore critico (vendor lock-in) |
| Latenza | Nessuna chiamata di rete | Round-trip di rete ad ogni imbedding, rilevante se la domanda deve essere incorporata in tempo reale |

Il vincolo di questo progetto (CPU locale, zero spese, dati non sensibili) esclude a monte le API
e restringe la scelta a modelli open-weights — ma è il vincolo del *nostro* esercizio, non una
regola generale: in un'azienda vera la tabella sopra è la decisione da prendere caso per caso, con
i numeri del proprio carico e budget, non con un'esclusione a priori.

## Usare più di un sistema di embedding nella stessa architettura

Due modi concreti in cui ha senso combinarli, non alternativi tra loro:

**Fusione dei punteggi (ensemble).** Interrogare più indici sulla stessa domanda — per esempio
BM25 + MiniLM + BGE, o due provider diversi — e unire le classifiche (es. con Reciprocal Rank
Fusion) invece di scegliere un solo metodo vincente. Aumenta il recall quando i metodi sbagliano in
punti diversi: BM25 perde i sinonimi, un embedding generico può perdere un termine tecnico raro
che BM25 troverebbe per corrispondenza esatta. Costo: più indici da costruire e mantenere, più
tempo per ogni domanda.

**Recupero in due stadi (retrieve-then-rerank).** Un modello leggero (MiniLM, o un'API economica)
recupera in fretta un ampio insieme di candidati su tutto il corpus; un modello più pesante, o un
cross-encoder dedicato al reranking (es. Cohere Rerank, BGE-reranker), rilegge solo quei candidati
e li riordina. Combina la velocità del primo stadio (il passaggio costoso lavora su decine di
candidati, non sull'intero corpus) con la qualità del secondo (il modello migliore giudica dove
conta di più: la cima della classifica). È il pattern più comune in produzione quando si vuole
qualità di punta senza pagare il costo di un modello pesante su ogni singolo documento del corpus.

**Quando non combinare:** se un singolo metodo soddisfa già la soglia di qualità richiesta, la
combinazione aggiunge solo complessità — più indici, più latenza — senza beneficio misurato. Va
giustificata da un confronto con numeri reali, non presunta a priori come "più metodi, meglio è".
