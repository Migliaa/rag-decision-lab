# Appunti 5 (importante) — Mappa delle scelte di design nel retrieval

Riassunto trasversale di Appunti1-4: non ripete le derivazioni (formule, esperimenti misurati
restano lì), raccoglie in un unico posto **ogni punto in cui esiste una vera scelta di design**
nella parte di retrieval di un RAG, le opzioni realmente usate in ambito professionale (locali e a
pagamento), e i criteri che spostano la scelta da un'opzione all'altra. Solo opzioni davvero
adottate nella pratica — niente scelte accademiche senza uso reale, tranne quelle che abbiamo
usato noi in questo progetto, sempre segnalate come tali quando non sono la scelta professionale
di punta. Indice: `INDICE_APPUNTI.md`.

Cinque decisioni, in ordine di dipendenza (la 1 va decisa prima delle altre, perché le altre
lavorano SUL RISULTATO della 1 — cambiare il chunking dopo obbliga a rifare tutto da capo):

1. Chunking — come tagliare i documenti in unità cercabili
2. Modello di embedding — come rappresentare quelle unità come vettori
3. Metodo di valutazione — come misurare se il retrieval funziona
4. Fusione di più sistemi di ricerca — come combinare classifiche diverse
5. Reranking — come raffinare i primi risultati con un modello più costoso

---

## 1. Chunking: come tagliare i documenti

**La domanda di design**: qual è l'unità atomica che verrà cercata e passata al generatore? Va
decisa per prima perché ogni cambiamento successivo (embedding, fusione, reranking) lavora SOPRA
il chunking già fatto — cambiarlo dopo significa ricalcolare tutto da zero (già discusso
all'inizio di questo studio).

| Strategia | Come funziona | Quando la sceglie un professionista |
|---|---|---|
| **Dimensione fissa con sovrapposizione (fixed-size + overlap)** | Taglio ogni N caratteri/token, con una sovrapposizione di M unità tra un pezzo e il successivo per non spezzare un'idea esattamente al confine | Baseline più comune in assoluto: semplice, deterministica, indipendente dal contenuto. È quella usata da IBM nel corpus MTRAG che stiamo usando noi (vedi Appunti1: gli id `documento-inizio-fine` lo dimostrano) |
| **Ricorsiva/gerarchica (recursive character/token splitting)** | Prova a tagliare prima ai confini di paragrafo, poi di frase, poi di parola, solo come fallback arriva al taglio a carattere fisso | Lo standard de facto nei framework professionali (LangChain, LlamaIndex lo offrono come default) quando i documenti hanno una struttura testuale normale (prosa, articoli) |
| **Consapevole della struttura del documento (structure-aware)** | Taglia ai confini reali del documento: intestazioni Markdown/HTML, righe di una tabella, sezioni di un PDF | Scelta quasi obbligata per documentazione tecnica, contratti, PDF con tabelle — tagliare a caso dentro una tabella o una sezione produce passaggi incomprensibili fuori contesto |
| **Semantica (semantic chunking)** | Incorpora frasi singole, taglia dove il significato cambia bruscamente (misurato come calo di similarità tra frasi consecutive) invece che a una lunghezza fissa | Quando un documento tratta più argomenti distinti senza una struttura tipografica che lo segnali — costo aggiuntivo: serve incorporare a livello di frase prima ancora di decidere i tagli |
| **Late chunking** (tecnica recente, 2024) | Si incorpora l'INTERO documento con un modello a contesto lungo, poi si derivano i vettori dei singoli chunk aggregando i vettori dei token nella zona di quel chunk — ogni chunk "eredita" un po' di contesto dell'intero documento invece di essere incorporato in isolamento | Quando molti chunk perdono senso letti da soli (es. "esso", "questo importo" senza sapere a cosa si riferiscono) — richiede un modello di embedding a contesto lungo (es. Nomic, Jina embeddings v3) |
| **Nessun chunking (documento intero come unità)** | Non si taglia affatto | Solo con documenti già brevi (poche centinaia di parole) o con embedding/generatori a contesto molto lungo — raro nel caso generale |

**Da evitare, non un'opzione professionale**: taglio a lunghezza fissa SENZA badare ai confini
(spezzare a metà parola o frase) e senza sovrapposizione — non è una scelta di design deliberata,
è un difetto di implementazione che si nota subito nei risultati (passaggi che iniziano/finiscono a
metà frase).

**In questo progetto**: usiamo il chunking già fatto da IBM per MTRAG (dimensione fissa a
carattere, con sovrapposizione — vedi l'esempio dei due passaggi del documento 10171 in Appunti1),
accettato come dato di partenza, non deciso da noi. È una scelta ragionevole e diffusa (prima riga
della tabella), non la più sofisticata disponibile.

---

## 2. Modello di embedding

**La domanda di design**: quale modello trasforma testo in vettori, per corpus e query. Ripreso e
consolidato da Appunti1.

| Famiglia | Esempi | Locale/API | Quando sceglierla |
|---|---|---|---|
| Bi-encoder generico piccolo | MiniLM (`all-MiniLM-L6-v2`) | Locale, gratuito | Prototipazione rapida su CPU, nessun vincolo di qualità stringente — **usato da noi in questo progetto** come primo termine di paragone, non perché sia la scelta professionale di punta |
| Bi-encoder specializzato per retrieval | BGE (`bge-small/base/large`), E5, GTE | Locale, gratuito | Scelta professionale di default quando serve restare locali/gratuiti: addestrati apposta per distinguere ruolo query/passaggio, misurabilmente migliori di un embedding generico sullo stesso compito (Appunti1/3) — **BGE-small usato da noi** |
| Contesto lungo | Nomic-embed-text-v1.5, Jina embeddings v3 | Locale (Nomic) o API (Jina) | Quando i chunk sono lunghi o serve "late chunking" (sezione 1) |
| API generiche di punta | OpenAI `text-embedding-3-large`, Cohere `embed-v3`, Google `text-embedding-004` | A pagamento | Quando serve la qualità più alta senza gestire infrastruttura locale, e i dati possono uscire dall'azienda — criterio completo in Appunti1 |
| API specializzata di dominio | Voyage AI `voyage-finance-2` (finanza), `voyage-law-2` (legale) | A pagamento | Quando il gergo di dominio è denso e un embedding generico perde precisione su quello specifico — la prima cosa che un'azienda con QUESTO stesso problema (dati finanziari) andrebbe a misurare |

Criterio di scelta riassunto (tabella completa in Appunti1): privacy/compliance e volume di
richieste spingono verso il locale; qualità di punta e assenza di manutenzione infrastrutturale
spingono verso l'API. Non esiste una risposta valida a prescindere dal contesto.

---

## 3. Metodo di valutazione del retrieval

**La domanda di design**: quale numero (o insieme di numeri) dice se il sistema funziona.
Consolidato da Appunti2/3.

| Metrica | Cosa cattura | Quando è la metrica giusta da guardare |
|---|---|---|
| **Recall@k** | Copertura: quota di oro trovata nei primi k, senza guardare la posizione | Quando perdere un documento rilevante è il rischio principale (es. due diligence, ricerca legale) — **usata da noi** |
| **Precision@k** | Quanto dei primi k è davvero rilevante | Quando il budget di contesto è stretto e ogni passaggio irrilevante incluso costa (sposta spazio a uno buono) |
| **nDCG@k** | Copertura E posizione insieme, in un solo numero | Standard de facto per confrontare configurazioni in automatico (scelta di chunking, embedding, fusione) — **usata da noi** |
| **MRR** (Mean Reciprocal Rank) | Quanto in alto sta il PRIMO risultato buono | Quando serve una sola fonte affidabile, non una rassegna (es. assistente che cita un solo documento) |
| **MAP** (Mean Average Precision) | Media di precision calcolata a ogni posizione rilevante | Benchmark IR storici (TREC); meno usata nei report moderni orientati al prodotto, dove nDCG/MRR sono preferiti per essere più diretti da spiegare |
| **Hit Rate@k** | Percentuale di domande con almeno un rilevante nei primi k (binario per domanda) | Sistemi conversazionali dove basta agganciare l'argomento giusto, il dettaglio arriva dopo |
| **RAGAS — faithfulness / answer relevancy / context precision-recall** | Non il retrieval puro: la risposta FINALE generata, e se è supportata dalle fonti | Quando esiste già un generatore a valle (D3/M2, non ancora affrontato in questo progetto) — misura un livello diverso, complementare, non sostitutivo di recall/nDCG |

**Rigore statistico, non solo la media**: con poche domande (il nostro pilota da 12) una
differenza tra due metodi può essere rumore, non un vero miglioramento. Il metodo professionale
è un test appaiato per domanda (es. Wilcoxon signed-rank, o bootstrap sulle differenze), non il
solo confronto delle medie — punto emerso nella conversazione su overfitting/dev-test split, non
ancora applicato in questo progetto.

---

## 4. Fusione di più sistemi di ricerca

**La domanda di design**: dato che BM25 ed embedding sbagliano in punti diversi (Appunti3), come
si combinano le loro classifiche in una sola. Consolidato e ampliato da Appunti4.

| Metodo | Come funziona | Uso professionale reale |
|---|---|---|
| **RRF (Reciprocal Rank Fusion)** | Somma `1/(c+rank)` per ogni sistema in cui un passaggio compare, ignora i punteggi grezzi | **Lo standard di fatto oggi**: implementato nativamente nella ricerca ibrida di Elasticsearch, Weaviate, Pinecone, OpenSearch. Non richiede normalizzare punteggi eterogenei (il problema pratico che risolve, vedi Appunti4) |
| **CombSUM / CombMNZ** | Somma (CombSUM) o somma moltiplicata per quante liste contengono il documento (CombMNZ) di punteggi NORMALIZZATI (es. min-max) | Storicamente diffuso nella IR classica, oggi scelto meno spesso di RRF perché richiede una normalizzazione affidabile tra sistemi eterogenei — resta valido quando i punteggi da fondere sono davvero comparabili (es. due embedding con coseno normalizzato, non BM25+coseno) |
| **RRF pesato** | Come RRF, ma con un peso diverso per sistema | Quando si ha evidenza misurata che un sistema è sistematicamente più affidabile di un altro sul proprio dominio, senza escludere gli altri |
| **Learning to rank** | Un modello (spesso gradient boosting) addestrato su dati etichettati per combinare più segnali (posizioni, punteggi, lunghezza, corrispondenze lessicali...) nel modo che massimizza nDCG | Scala produttiva reale con grandi volumi di query etichettate (log utenti, non 12 domande di pilota) — motori di ricerca commerciali maturi la usano quasi tutti, un prototipo no |

**In questo progetto**: non ancora implementato (prossimo passo naturale dopo il corpus completo).
Per lo scenario misurato in Appunti3 (BM25 eterogeneo rispetto agli embedding), RRF è la scelta
professionale coerente — è quella che verrà provata per prima.

---

## 5. Reranking: quale modello per il secondo stadio

**La domanda di design**: dopo la fusione (sezione 4), quale modello rilegge i primi N candidati
per il riordino finale. Consolidato da Appunti4.

| Approccio | Architettura | Prodotti/modelli reali | Quando sceglierlo |
|---|---|---|---|
| **Cross-encoder** | Query e passaggio concatenati, un solo passaggio nel modello per coppia — massima precisione, non pre-calcolabile | `cross-encoder/ms-marco-MiniLM-*` (open, sentence-transformers), BGE-reranker-base/large/v2-m3 (open, multilingue) | Scelta di default per il reranking quando si gestisce l'infrastruttura in locale |
| **Reranker via API** | Stesso principio, offerto come servizio gestito | Cohere Rerank, Voyage `rerank-2`, Jina Reranker | Quando si vuole evitare di ospitare il modello, o si vuole il punteggio di qualità più alto disponibile senza gestirlo |
| **Late interaction (via di mezzo)** | Ogni token di query e passaggio ha un proprio vettore precalcolabile (come un bi-encoder), ma il punteggio finale confronta i vettori token-per-token invece di un solo vettore aggregato — più preciso di un bi-encoder puro, più veloce di un cross-encoder perché i vettori dei passaggi restano precalcolati | ColBERT / ColBERTv2 (open), usato in produzione da motori come Vespa | Quando il costo di un cross-encoder puro su molti candidati è comunque troppo alto, ma un bi-encoder semplice non basta |

**In questo progetto**: non ancora implementato — sarà il passo dopo la fusione RRF, applicato solo
ai primi 20-100 candidati fusi (mai sull'intero corpus, per il motivo di costo spiegato in
Appunti4).

---

## Come si incastrano: l'ordine delle decisioni in una pipeline reale

```
1. Chunking (deciso una volta, tutto il resto dipende da questa scelta)
      ↓
2. Uno o più modelli di embedding + BM25 in parallelo (indicizzazione)
      ↓
4. Fusione (RRF o simili) delle classifiche prodotte al passo 2
      ↓
5. Reranking (cross-encoder o via API) SOLO sui primi N fusi
      ↓
3. Valutazione (recall/nDCG/precision, poi RAGAS quando c'è un generatore) — non è uno stadio
   della pipeline, è la misura che guida OGNI decisione sopra, ripetuta ad ogni cambiamento
```

Nessuna di queste scelte è indipendente dal contesto: uno stesso progetto in un'azienda diversa
(più dati, budget diverso, requisiti di privacy diversi) può ragionevolmente arrivare a una
combinazione diversa da quella che sceglieremo qui — il criterio, ripetuto in ogni Appunti di
questo studio, resta motivare ogni scelta con compito, qualità misurata e risorse disponibili, non
con la fama del nome.
