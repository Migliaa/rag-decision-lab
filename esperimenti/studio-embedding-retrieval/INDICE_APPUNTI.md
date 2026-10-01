# Indice degli Appunti — studio embedding e retrieval

**Da dove cominciare a leggere**: [Appunti10.md](Appunti10.md) è il percorso decisionale della
sezione e tiene insieme tutte le altre note. Le note 3, 6 e 7 sono le tappe del percorso, riscritte
con il senno di poi il 20 settembre 2026: contengono i numeri misurati allora accanto a quelli
corretti, e per ogni decisione presa dicono se è rimasta o è stata rovesciata. La misura valida del
progetto è quella di [Appunti8.md](Appunti8.md), su 180 domande.

Convenzione: ogni `AppuntiN.md` è una nota di studio autonoma, associata a una o più figure
`FigN.X` (etichetta di citazione, non un rinominare dei file immagine — i file restano con il nome
prodotto dallo script che li genera, elencato qui sotto per ognuno). Aggiornare questa tabella ogni
volta che nasce un nuovo Appunti o una nuova figura numerata.

| Appunti | Argomento | Figure | File immagine | Script che le genera |
|---|---|---|---|---|
| [Appunti1.md](Appunti1.md) | Scelta del modello di embedding (MiniLM vs BGE, alternative di mercato, combinare più sistemi) | Fig1.1 | `output/embedding_2d_tsne_categorie.png` | `02b_visualizza_embedding_tsne.py` |
| [Appunti2.md](Appunti2.md) | Retrieval da una query, recall@k e nDCG@k, altre metriche professionali | Fig2.1 | `output/query_vicini.png` | `03_query.py`, `04_figura_query_vicini.py`, `05_metriche_singola_query.py` |
| [Appunti3.md](Appunti3.md) | **Tappa 1 del percorso**: la prima misura e perché ha ingannato — BM25 vs MiniLM vs BGE, pilota contro corpus completo, come si leggono recall e nDCG, le tre decisioni prese e il loro destino | Fig3.1, Fig3.2 | `output/metriche_12_domande.png`, `output/metriche_corpus_completo.png` | `06_metriche_12_domande.py`, `07_metriche_corpus_completo.py` |
| [Appunti4.md](Appunti4.md) | Teoria (non ancora misurata): RRF in dettaglio, reranking, architettura RAG (contesto/citazioni/errori a monte), agenti di ricerca, wiki compilata da LLM | nessuna (nota solo teorica) | — | — |
| [Appunti5-importante.md](Appunti5-importante.md) | **Riassunto trasversale**: mappa di tutte le scelte di design del retrieval (chunking, embedding, valutazione, fusione, reranking) con opzioni professionali reali e criteri di scelta | nessuna (riassume Appunti1-4) | — | — |
| [Appunti6.md](Appunti6.md) | **Tappa 2**: fondere le classifiche e scegliere la profondità — tre schemi di pesi a confronto, nascita della distinzione fra tetto e conversione, la decisione sulla lunghezza della lista poi rovesciata | Fig6.1 | `output/fusione_rrf.png` | `08_fusione_rrf.py` |
| [Appunti7.md](Appunti7.md) | **Tappa 3**: il riordino con cross-encoder — bi-encoder contro cross-encoder, il tetto fissato dal recupero a monte, il guadagno medio che nasconde casi peggiorati | Fig7.1 | `output/reranking_bge_large.png` | `09_reranking.py` |
| [Appunti8.md](Appunti8.md) | **Tappa 4, la misura valida**: il campione da 12 domande invalidava le tappe precedenti, misura di riferimento su 180 domande, chunking scagionato, annotazioni incomplete, strategie di reranking a confronto, multi-formulazione; in coda, la fusione di reranker di famiglie diverse verificata con confronto appaiato | Fig8.1–8.5 | `output/baseline_180.png`, `output/strategie_reranking_complete.png`, `output/multiquery.png`, `output/configurazione_finale.png`, `output/strategie_con_mxbai.png` | `10_baseline_180.py`, `11`+`11b`+`12b`, `14_multiquery.py`, `16_configurazione_finale.py`, `19_strategie_con_mxbai.py`, `20_significativita_fusione_reranker.py` |
| [Appunti9.md](Appunti9.md) | **Tappa 5, il banco decisioni**: lo strumento con cui le scelte diventano confrontabili, criteri propri per ogni decisione, evidenza dichiarata riga per riga, vincoli applicativi | Fig9.1–9.4 | `output/banco_intero.png`, `output/banco_unita_indicizzazione.png`, `output/banco_reranker.png`, `output/banco_reranker_vincolato.png` | `strumenti/banco-decisioni-rag/`, `18_screenshot_banco.py` |
| [Appunti10.md](Appunti10.md) | **Il percorso decisionale di M1**: raccordo fra tutte le tappe, come è cambiato il criterio di decisione, i tre errori e cosa li ha resi possibili, dove il banco entra e dove no | nessuna (rimanda alle figure delle tappe) | — | — |
| [Appunti11.md](Appunti11.md) | **Il test a pagamento**: progetto dell'esperimento non ancora eseguito, costo misurato sul corpus reale (~1,30 $ in tutto), cosa dimostrerebbe, e le modifiche al banco che la sua progettazione ha reso necessarie | nessuna (nota di progetto) | — | — |

Figure prodotte ma non (ancora) associate a un Appunti numerato, materiale esplorativo intermedio:

- `output/embedding_2d_pca.png` — primo tentativo (PCA + cluster KMeans non verificati), superato
  da Fig1.1 che usa argomenti reali dalle qrels invece di cluster automatici.

Non ancora scritto: nota sulla misura a scala reale (corpus FiQA/MTRAG completo invece del pilota a
512 passaggi). La nota sul chunking non serve più: l'ipotesi è stata verificata e archiviata in
Appunti8 con un conteggio diretto (1,06 passaggi per documento).
