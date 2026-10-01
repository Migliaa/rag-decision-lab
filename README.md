# Scegliere un sistema di ricerca con i dati

Studio su come scegliere e motivare i componenti di un sistema RAG (ricerca di brani su documenti finanziari e risposta di un modello). Il prodotto principale è il metodo, non il punteggio: un **banco delle decisioni**, una pagina HTML senza server in cui ogni componente (unità di indicizzazione, modello di embedding, tipo di ricerca, riordinatore, generatore) ha la sua tabella di alternative con i criteri che contano per *quella* decisione, e dove i vincoli del contesto (dati riservati, budget zero, meno di un secondo per domanda, solo CPU, alto volume) si attivano come filtri ed escludono le opzioni incompatibili. Ogni valore dichiara se è misurato qui, dichiarato da un fornitore o non rilevato, e solo i misurati possono essere evidenziati come scelta migliore.

**Banco online:** https://rag-decision-lab-banco.vercel.app · le misure che lo alimentano vengono da un esperimento reale su recupero, generazione e un agente minimo, descritto sotto.

![Il banco delle decisioni: vincoli attivabili in alto, una tabella di alternative per ogni componente](report/figure/fig-0-banco.png)

## Come ragiona il banco

- **Un criterio diverso per ogni decisione.** L'unità di indicizzazione non si sceglie guardando il nDCG, che dipende da tutta la pipeline a valle, ma da quante unità produce l'indice, quanto sono lunghe e quante risposte restano spezzate; il modello di embedding da qualità, costo e dimensione dell'indice. La prima versione, con una griglia uniforme, è stata scartata.
- **Il contesto decide.** Attivando insieme «dati riservati» e «budget zero», delle quattordici righe del modulo riordinatore ne restano otto, e spariscono tutte le migliori per latenza. Con un altro contesto (dati non riservati, budget per le API) tornano in gioco servizi commerciali come Cohere Rerank, che il banco registra con il prezzo (2 $ per mille interrogazioni) e come non provati.
- **I buchi restano buchi.** Un valore non rilevato resta vuoto invece di essere omesso o stimato.

## Provare

```bash
uv sync                                   # Python 3.14; per la parte di generazione: uv sync --extra generazione
uv run python scripts/scarica_dati.py     # scarica MTRAG/FiQA dalla revisione congelata e verifica gli hash
uv run python -m unittest discover tests  # test delle funzioni di valutazione e di tokenizzazione
uv run python scripts/prepare_fiqa_pilot.py && uv run python -m scripts.run_d1_pilot   # pilota: 12 domande, 512 brani
start strumenti/banco-decisioni-rag/banco_decisioni_rag.html   # il banco (su macOS/Linux: open / xdg-open)
```

Le misure sulle 180 domande (`esperimenti/studio-embedding-retrieval/06…20`) indicizzano 61.022 brani su CPU: circa un'ora e mezza per due modelli, e i punteggi dei riordinatori da ore a giorni. I risultati già calcolati sono salvati in `esperimenti/*/output/` (JSON e grafici), ma le matrici di embedding (`.npy`, alcuni GB) no. La generazione e l'agente usano l'API di Gemini: serve `esperimenti/studio-generazione/apikey.env` con `GEMINI_API=<chiave>` (vedi `apikey.env.example`); il pilota di generazione costa pochi centesimi.

## Che cosa hanno mostrato le misure

Le misure usano solo modelli gratuiti, in locale su CPU, quindi i valori assoluti sono modesti: il valore del lavoro sta nel modo in cui sono state confrontate le alternative. Recupero, su 180 domande di MTRAG (parte FiQA, 61.022 brani), una sola esecuzione per configurazione:

| configurazione | recall@10 | nDCG@10 |
|---|---|---|
| `bge-small-en-v1.5` da solo (punto di partenza) | 0.417 | 0.332 |
| + riordino con `ms-marco-MiniLM-L-6-v2` sui primi 30-50 candidati | 0.478 | 0.387 |
| + candidati costruiti con `gte-base-en-v1.5` | 0.489 | 0.395 |

Il massimo raggiungibile con 200 candidati è 0.82 di recall, quindi il sistema ne converte meno dei due terzi. Gli intervalli di confidenza al 95% di ogni riga sono larghi circa ±0.05; l'aumento dovuto al riordino non è stato verificato con un test appaiato e va letto come indicativo, mentre la fusione fra riordinatori di famiglie diverse (+0.028 nDCG@10, Wilcoxon p = 0.005) sì, ma costa venti minuti a domanda su CPU.

## Che cosa fa

- **Problema.** Ogni componente di un RAG ha alternative ragionevoli (modello di embedding, tipo di ricerca, riordinatore, generatore) e la scelta dipende da costo, hardware, velocità e riservatezza dei dati; i valori di una classifica pubblica non si trasferiscono automaticamente.
- **Per chi.** Per chi deve scegliere e motivare un sistema di ricerca su documenti, non per chi cerca un prodotto pronto.
- **Entra.** Il corpus FiQA di MTRAG (forum di finanza personale, [IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark), revisione `2c618bb`), le domande dell'ultimo turno e le fonti annotate a mano.
- **Esce.** Misure per configurazione (recall@10, nDCG@10, intervalli, tempi), grafici, letture a mano di casi, e il banco delle decisioni con le opzioni e i vincoli.

## Che cosa è emerso

- **Recupero.** Il guadagno quasi tutto viene dal riordino con un cross-encoder. Le misure sulle sole 12 domande del pilota ordinavano i metodi all'opposto di quelle sulle 180: ogni confronto è stato rifatto su tutte. Allargare la lista di candidati oltre 50 peggiora il risultato (più candidati, più occasioni di errore del riordinatore). Nove leve non hanno reso nulla: diversificazione dei risultati, interpolazione dei punteggi, fusione di riordinatori della stessa famiglia, riordinatore grande (`bge-reranker-large`: +0.006 nDCG per venti volte il costo), multi-formulazione sopra il riordino, modello di embedding più grande, quantizzazione a 8 bit, lista più lunga, cascata fra modelli simili.
- **Le classifiche pubbliche non si trasferiscono.** `gte-base-en-v1.5` fa 0.487 su BEIR/FiQA contro 0.403 di `bge-small`, ma su queste domande conversazionali 0.300 contro 0.332 (nDCG@10).
- **Generazione** (pilota su 5 domande con `gemini-3.1-pro-preview`, giudicate a mano). Il recupero si misura per brano, la risposta per domanda: per 136 domande su 180 almeno una fonte annotata è nel contesto, per 44 nessuna. Un quarto modo di fallire non previsto è un contesto pertinente per argomento ma fatto di frammenti di discussione senza risposta. Nelle domande in cui la configurazione scelta e quella semplice producono contesti diversi (campione di 6), la scelta, migliore in media, peggiora quasi un terzo dei casi: la differenza è nella completezza della risposta, non fra giusta e sbagliata.
- **Agente** (ciclo con due strumenti, cerca e leggi, limite di sei passi, `gemini-3.1-pro-preview`, due domande). Sulla prima, vicino al limite ha prodotto una risposta sicura e infondata dopo aver speso il budget a riformulare la domanda; la seconda è rimasta bloccata oltre venti minuti su una singola chiamata, perché il limite era sui passi e non sul tempo. Serve un tetto su entrambi, e così com'è non è un sistema da mettere davanti a un utente.
- **Wiki compilata da un modello**: analizzata, non costruita.

## Limiti

Un solo corpus e un solo benchmark, annotazioni incomplete (circa 3 fonti per domanda su 61.022 brani: ogni punteggio assoluto è sottostimato di una quantità ignota) e domande riscritte da un annotatore umano, che in un caso letto a mano sposta l'argomento. Ogni configurazione è una sola esecuzione. La parte di generazione e quella dell'agente sono pilota su 5 e 2 domande: indicano un modo di fallire, non misurano una frequenza. Il test a pagamento dei servizi di embedding e riordino (Appunti11, circa 1,30 $) è progettato e non eseguito, e le alternative locali per il generatore sono nel banco come non misurate. Il codice è pensato come quaderno di esperimenti numerati (uno script per passo, mai modificato dopo l'esecuzione), non come libreria.

## Dove sono i file

| Cosa | Dove |
|---|---|
| Report (testo e figure) | `report/report.md`, `report/figure/` |
| Banco delle decisioni | `strumenti/banco-decisioni-rag/` (`CONTESTO_BANCO.md` spiega regole e codice) |
| Note di studio, dalla più utile: percorso decisionale, misura valida, banco, test a pagamento | `esperimenti/studio-embedding-retrieval/Appunti10.md`, `Appunti8.md`, `Appunti9.md`, `Appunti11.md` (indice in `INDICE_APPUNTI.md`) |
| Script del recupero, numerati | `esperimenti/studio-embedding-retrieval/01…20_*.py` |
| Generazione e agente | `esperimenti/studio-generazione/`, `esperimenti/studio-agenti/` |
| Recupero di base, pilota, test | `src/retrieval.py`, `scripts/`, `tests/`, `laboratorio/` |
| Provenienza dei dati | `data/manifest.json`, `laboratorio/DATI.md` |

MIT, vedi [`LICENSE`](LICENSE).
