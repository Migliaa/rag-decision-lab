# Appunti 8 — Revisione critica del sistema: cosa era rotto, cosa no, e dove sta il margine

Figure di riferimento: **Fig8.1** = `output/baseline_180.png` (script `10_baseline_180.py`);
**Fig8.2** = `output/strategie_reranking_complete.png` (script `11_punteggi_reranker.py`,
`11b_punteggi_reranker_leggeri.py`, `12b_strategie_reranking_complete.py`); **Fig8.3** =
`output/multiquery.png` (script `14_multiquery.py`); **Fig8.4** = `output/configurazione_finale.png`
(script `16_configurazione_finale.py`). Dati: `output/baseline_180.json`,
`output/strategie_reranking_complete.json`, `output/multiquery.json`,
`output/configurazione_finale.json`. Indice: `INDICE_APPUNTI.md`.

Questa nota nasce da una richiesta di verifica: i risultati di Appunti3-7 erano peggiori di quanto
il progetto lasciasse sperare, e l'ipotesi scritta lì — la colpa è del chunking a monte — andava
controllata invece che assunta. Il controllo ha trovato un errore, ma non quello che cercavamo.

## 1. L'errore vero: dodici domande non sono un campione

Tutte le misure da Appunti3 in poi usavano le 12 domande del pilota. Ripetendo esattamente le stesse
misure sulle 180 domande che il dataset MTRAG mette a disposizione — costo: qualche minuto, perche'
gli embedding del corpus erano gia' calcolati e servivano solo le domande nuove — i valori cambiano
di due o quattro volte, e cambia l'ordine dei metodi.

| metodo | recall@10 su 12 domande | recall@10 su 180 domande | intervallo di confidenza 95% |
|---|---|---|---|
| BM25 | 0.056 | **0.211** | 0.166 – 0.260 |
| MiniLM | 0.264 | **0.369** | 0.320 – 0.422 |
| BGE-small | 0.167 | **0.417** | 0.365 – 0.469 |
| RRF uniforme | 0.264 | **0.403** | 0.354 – 0.455 |

Sul campione da 12, BGE-small sembrava il peggiore dei tre metodi; su 180 e' il migliore, come dicono
i benchmark pubblici (su BEIR/FiQA `bge-small-en-v1.5` riporta nDCG@10 di 0.403 contro 0.369 di
`all-MiniLM-L6-v2`). La sottostima inoltre non era uniforme — quattro volte per BM25, due volte e
mezzo per BGE, una volta e mezzo per MiniLM — quindi non si trattava di una scala sbagliata ma di un
riordinamento: il campione sceglieva un vincitore diverso da quello reale.

La conseguenza va detta senza attenuazioni: la conclusione centrale di Appunti3 («sul corpus completo
tutti i metodi collassano, BGE finisce peggio di MiniLM») era falsa, e su quella conclusione sono
state costruite le scelte di Appunti6 e Appunti7. Ho messo un avviso di correzione in testa alle tre
note invece di riscriverle, perche' l'errore e il modo in cui e' stato scoperto valgono piu' della
versione ripulita.

L'intervallo di confidenza nella tabella e' il motivo per cui la cosa era prevedibile: con 180
domande l'incertezza sulla media e' di circa ±0.05, con 12 domande sarebbe stata di circa ±0.20 —
larga quanto l'intera differenza fra il metodo migliore e il peggiore. Il campione non era
sfortunato, era inadeguato in partenza, e l'avvertimento generico sull'overfitting che avevamo
scritto piu' volte non era stato tradotto nell'unica azione che serviva, cioe' misurare l'incertezza.

## 2. Cosa invece era sano

Sono state controllate, una per una, le cause tipiche di risultati artificialmente bassi:

| controllo | esito |
|---|---|
| giudizi di rilevanza con punteggio 0 contati come oro | assente: tutti i 535 giudizi hanno punteggio 1 |
| embedding non normalizzati (prodotto scalare che non e' coseno) | assente: norma L2 = 1.000000 su tutti i campioni |
| righe corrotte dalle interruzioni per batteria | assente: nessuna riga a norma nulla |
| passaggi oro mancanti dal corpus | assente: tutti presenti |
| prefisso di BGE applicato per errore ai passaggi | assente: usato solo sulle domande, come prescritto |

Tre difetti minori reali, invece, esistono. MiniLM ha una finestra di 256 token e il 13% dei passaggi
la supera, quindi di quelli legge solo l'inizio; BGE-small ne ha 512 e taglia molto meno. Quanta
parte del divario fra i due modelli venga dal troncamento e quanta dalla qualita' della
rappresentazione non e' stato misurato, e non va dato per scontato. Il nostro BM25
usa una tokenizzazione senza riduzione alla radice, quindi "taxes" e "taxation" restano parole
diverse. E la scelta della formulazione della domanda pesa quanto la scelta del modello: usando la
variante `fiqa_questions`, che concatena tutti i turni della conversazione, il recall@10 di BGE
crolla a 0.196 contro lo 0.417 della variante riscritta — il rumore dei turni precedenti danneggia
piu' di quanto il contesto aiuti.

## 3. Il chunking non e' colpevole

L'ipotesi scritta in Appunti6 e Appunti7 era che i tagli dei passaggi ereditati da IBM spezzassero le
risposte e mettessero un tetto a tutto il resto. I dati dicono il contrario: il corpus ha 57.638
documenti distinti per 61.022 passaggi, cioe' **1,06 passaggi per documento**, e solo il 4,2% dei
documenti e' diviso in piu' di un pezzo. Inoltre, per ognuna delle domande esaminate, i passaggi oro
stanno in documenti diversi fra loro, quindi non esiste nessuna risposta tagliata a meta' da
recuperare unendo pezzi adiacenti. L'accusa era un'ipotesi plausibile mai verificata, e verificarla
costava una riga di conteggio.

## 4. Un limite che nessun numero rivela: le annotazioni sono incomplete

Leggendo i primi risultati che il sistema restituisce e che la valutazione conta come errori, si vede
un problema che le metriche non possono mostrare. Per la domanda *"Should I consider investing in
existing startups instead of starting my own?"* i primi tre risultati non annotati discutono tutti
esattamente quel dilemma; per *"Can I get my credit report in the US?"* un risultato non annotato
spiega che in alcuni stati si ha diritto a un rapporto gratuito all'anno, cioe' risponde. Nessuno dei
due conta come successo.

FiQA e' un forum finanziario: molte risposte diverse trattano lo stesso tema in modo ugualmente
valido, ma ne sono annotate in media 2,97 per domanda su 61.022 passaggi. Ne seguono tre conseguenze
pratiche:

- I valori assoluti sottostimano la qualita' reale del sistema. Un recall@10 di 0.48 non significa
  che in meta' dei casi il contesto non contiene una risposta.
- Il confronto fra metodi resta valido, perche' la sottostima colpisce tutti allo stesso modo.
- Le strategie che premiano la varieta' delle risposte sono penalizzate piu' delle altre, perche'
  recuperano proprio il materiale valido che l'annotazione non copre. Un risultato negativo sulla
  diversificazione, in questo dataset, va letto con questa riserva.

Per un sistema RAG reale la domanda giusta non e' «ho recuperato il documento annotato» ma «il
contesto contiene abbastanza per rispondere»: e' la differenza fra le metriche di retrieval e le
metriche di tipo RAGAS gia' descritte in Appunti2, e qui si vede perche' esistono.

## 5. Dove sta davvero il margine

Il reranking non puo' promuovere cio' che non e' nella lista di candidati, quindi il numero da
guardare e' quanto oro entra nella lista, per ogni profondita':

| candidati | oro presente nella lista | recall@10 ottenuto riordinando | quota del disponibile convertita |
|---|---|---|---|
| 10 | 0.403 | 0.403 (nessun riordino possibile) | — |
| 50 | 0.687 | 0.478 | 70% |
| 100 | 0.768 | 0.470 | 61% |
| 200 | 0.827 | 0.459 | 56% |

Allargare la lista alza il tetto e abbassa la resa: con piu' candidati plausibili davanti, il
cross-encoder sbaglia di piu' e ne porta in cima una frazione minore. Il massimo del prodotto sta
intorno a 50, il che contraddice la scelta di k=100 fissata in Appunti6 — scelta presa, va ricordato,
sulle 12 domande.

Il limite non e' quindi la profondita' della lista ma la capacita' di discriminazione del modello che
la riordina: per avvicinarsi a 0.83 serve un reranker migliore, non una lista piu' lunga.

## 6. Un vincolo di calcolo che decide il progetto

`bge-reranker-large`, il reranker piu' forte fra quelli gratuiti e quello scelto in Appunti6, su
questa CPU impiega **2,7 minuti per domanda** con 100 candidati: otto ore per una singola valutazione
su 180 domande, e la valutazione va rifatta a ogni variante di strategia. `ms-marco-MiniLM-L-6-v2`,
che ha 22 milioni di parametri contro 560, impiega 9 secondi per domanda con 200 candidati — due
ordini di grandezza in meno.

La via d'uscita standard per l'inferenza su CPU — la quantizzazione dinamica a 8 bit, che sostituisce
i pesi in virgola mobile con interi e di solito accelera due o tre volte — qui e' stata provata e
**rallenta**: 0.84 volte la velocita' originale, con fedelta' degli embedding perfetta (similarita'
coseno 1.00000 con la versione non quantizzata). Non e' un problema di qualita' ma di supporto: senza
il backend ottimizzato per questa combinazione di processore e libreria, il costo di conversione
supera il risparmio. Vale come promemoria che le ottimizzazioni note vanno misurate sulla macchina
che si ha, non assunte dalla documentazione.

Il modello grande e' stato quindi escluso, e non perche' sia peggiore. E' la forma concreta di un
compromesso che in un contesto professionale si risolve diversamente: con una GPU il divario di costo
si riduce a un fattore piccolo e il modello grande diventa la scelta ovvia; senza, la scelta e'
obbligata verso il modello piccolo. Vale la pena registrare l'ordine di grandezza — un cross-encoder
valuta ogni coppia domanda-passaggio con un passaggio completo della rete, quindi il costo cresce col
prodotto (numero di domande × numero di candidati), mentre un bi-encoder paga il corpus una volta
sola e poi confronta vettori.

## 7. Le quattro leve chieste, e cosa dicono le misure

**Profondita' della lista.** Il massimo sta a **30 candidati**, non a 50 ne' a 100: 0.487 di
recall@10 contro 0.478 a 50 e 0.470 a 100. La lista va dimensionata sulla precisione del reranker
disponibile, non sul tetto teorico che si vorrebbe raggiungere.

**Un reranker piu' grande.** `ms-marco-MiniLM-L-12-v2` ha una volta e mezzo i parametri della
versione a 6 strati e non la batte (0.484 contro 0.487 nel confronto migliore-contro-migliore, con
intervalli di confidenza ampiamente sovrapposti). Piu' capacita' nella stessa famiglia e con lo
stesso addestramento non produce piu' qualita'.

**Fondere due reranker.** RRF fra i due modelli sui primi 50 da' il miglior recall@10 di tutte le
strategie provate, 0.494, ma con lo stesso nDCG@10 del reranker singolo (0.389) e un margine di
0.007 dentro l'incertezza del campione. La spiegazione e' nella scelta dei modelli: i due vengono
dalla stessa famiglia e dallo stesso addestramento su MS MARCO, quindi sbagliano sulle stesse
domande, e fondere due liste che sbagliano insieme non aggiunge informazione. Il tentativo di
includere un terzo modello di famiglia diversa (`mxbai-rerank-xsmall`, basato su DeBERTa-v3) e'
stato interrotto: 20 minuti per domanda su questa CPU, sessanta ore per una valutazione.

**Cascata.** Filtrare con il modello economico e riordinare i sopravvissuti con
`ms-marco-MiniLM-L-12-v2` **peggiora** il risultato (0.463 contro 0.487), qualunque sia il numero di
sopravvissuti: una cascata non migliora per il fatto di essere una cascata, serve che il secondo
modello sia davvero migliore del primo.

Che sia esattamente questa la condizione lo conferma la prova opposta. Ridurre i sopravvissuti a 20
rende economicamente possibile usare `bge-reranker-large`, il modello che a 100 candidati costava
otto ore: a 20 ne costa una e mezza, ed e' il motivo per cui la cascata esiste come tecnica.

| | recall@10 | nDCG@10 |
|---|---|---|
| solo reranker economico, primi 100 | 0.470 | 0.381 |
| cascata: economico su 100, `bge-reranker-large` sui primi 20 | **0.495** | **0.397** |

A parita' di tutto il resto il modello grande guadagna davvero, 0.025 di recall e 0.016 di nDCG. Il
confronto onesto, pero', non e' questo ma quello con la migliore configurazione economica gia'
disponibile (0.489 / 0.394): li' il vantaggio scende a 0.006, dentro l'incertezza del campione, ed e'
pagato con un fattore venti nel costo di calcolo. Il modello grande e' migliore, e su questo corpus
non abbastanza da giustificarsi senza una GPU.

**Interpolare invece di sostituire.** L'idea era conservare l'informazione del recupero quando il
cross-encoder sbaglia, combinando i due punteggi come α·reranker + (1−α)·recupero. Scegliendo α sulle
domande pari e verificando su quelle dispari, il valore selezionato e' **α = 1.0**: il punteggio di
prima fase non aggiunge nulla oltre a cio' che il reranker gia' sa. Il risultato e' negativo e va
riportato come tale; la strategia resta utile in situazioni diverse da questa, per esempio quando il
recupero porta un segnale che il reranker non puo' vedere (freschezza, autorevolezza della fonte,
appartenenza a una sezione giusta del corpus).

**Diversificare i risultati (MMR).** Penalizzare i candidati troppo simili fra loro non migliora
nulla qui: con penalizzazione forte (λ=0.5) il risultato peggiora sensibilmente, con penalizzazione
debole (λ=0.9) resta identico al riordino puro (0.487 contro 0.487). Due spiegazioni concorrono, e la seconda e'
la piu' importante: il corpus ha poca ridondanza reale (un passaggio per documento), e soprattutto le
annotazioni incomplete descritte al punto 4 penalizzano proprio le risposte alternative che la
diversificazione va a cercare. Su questo dataset la misura non e' conclusiva contro la tecnica.

**Loop di reranking.** Riapplicare lo stesso modello agli stessi candidati non produce nulla: e'
deterministico, il secondo giro restituisce l'ordine del primo. Il concetto ha senso solo se ogni
giro cambia un ingresso, e allora prende due forme distinte, con proprieta' molto diverse:

- la *cascata* — modello economico su lista larga, modello costoso sui pochi sopravvissuti — che
  riordina soltanto, quindi resta sotto il tetto della lista iniziale;
- il *ritorno al corpus* — i primi risultati modificano la domanda e si cerca di nuovo su tutti i
  61.022 passaggi — che e' l'unica forma capace di alzare il tetto, perche' aggiunge candidati invece
  di riordinarli.

La distinzione e' la cosa da portarsi via: una catena di riordinamenti, per quanto lunga e costosa,
non puo' superare il limite fissato dal recupero iniziale.

## 8. Cosa e' disponibile fuori da qui, gratuito e non

Il divario fra i due modelli di embedding provati (0.369 contro 0.417) e' il piu' grande osservato
fra tutte le leve, e i due modelli sono entrambi fra i piu' piccoli in circolazione. Sul benchmark
BEIR/FiQA — lo stesso dominio, con domande non conversazionali, quindi indicativo e non trasferibile
alla lettera — i valori pubblicati di nDCG@10 sono:

| modello di embedding | parametri | nDCG@10 su FiQA | licenza |
|---|---|---|---|
| all-MiniLM-L6-v2 (usato qui) | 22M | 0.369 | gratuito |
| bge-small-en-v1.5 (usato qui) | 33M | 0.403 | gratuito |
| e5-base-v2 | 109M | 0.399 | gratuito |
| bge-base-en-v1.5 | 109M | 0.406 | gratuito |
| **gte-base-en-v1.5** | 137M | **0.487** | gratuito |
| Qwen3-Embedding-0.6B | 600M | fra i primi su MTEB | gratuito |
| Voyage `voyage-finance-2` | non dichiarati | specializzato sul dominio finanziario | a pagamento |
| Cohere `embed-v3` | non dichiarati | con distinzione query/documento nativa | a pagamento |

`gte-base-en-v1.5` e' l'anomalia della tabella: quattro volte i parametri di bge-small per venti
punti percentuali di guadagno relativo, gratuito, e con una finestra di 8192 token che eliminerebbe
anche il troncamento descritto al punto 2.

**Ed e' risultato inutilizzabile**, per un motivo che vale la pena registrare perche' si ripresenta
ovunque. Il modello richiede `trust_remote_code=True`: non usa un'architettura standard della
libreria, porta con se' il proprio codice, che viene eseguito al caricamento. Quel codice qui si
rompe in due punti — registra un buffer di indici come non persistente, e con il caricamento a basso
consumo di memoria ormai predefinito quel buffer viene materializzato da memoria non inizializzata
(valori come 2738746490880 usati come indici di posizione); reinizializzandolo a mano si arriva alla
seconda rottura, una chiamata a un metodo che la versione di `transformers` installata non espone
piu'. La questione non e' la fiducia nell'autore, che e' il modo in cui di solito si presenta quella
opzione: e' che il codice del modello invecchia separatamente dalla libreria che lo ospita, quindi un
modello con architettura standard resta utilizzabile per anni mentre uno con codice proprio smette di
funzionare quando la libreria cambia — e il conto arriva nel momento in cui serve.

Il ripiego e' `bge-large-en-v1.5`: architettura BERT standard, stessa famiglia e stesse convenzioni
di bge-small gia' in uso, su FiQA circa 0.45 contro 0.403. Costo misurato su questa macchina: fra le
dieci e le tredici ore per incorporare i 61.022 passaggi, con checkpoint per sopravvivere agli
spegnimenti.

Sul fronte dei reranker, i modelli forti gratuiti (`bge-reranker-v2-m3`, `Qwen3-Reranker-0.6B`,
`mxbai-rerank-large`) condividono il problema di `bge-reranker-large`: sono inutilizzabili su CPU per
un ciclo di esperimenti. I servizi a pagamento (Cohere Rerank, Voyage `rerank-2`) risolvono
esattamente questo, vendendo capacita' di calcolo per coppia valutata — che e' il motivo per cui
esistono come prodotto separato, e non un dettaglio commerciale: la struttura di costo del
cross-encoder e' intrinsecamente quella.

Due indicazioni pratiche ricorrenti nella letteratura applicativa meritano di essere provate qui,
perche' costano poco: interrogare il corpus con piu' formulazioni della stessa domanda e fondere le
liste (e' il punto 9), e mantenere la lista di reranking corta — l'indicazione piu' citata e' 50
candidati in ingresso e una quindicina in uscita, coerente con quanto misurato al punto 5.

## 9. Interrogare con piu' formulazioni: l'unica leva che alza il tetto (Fig8.3)

L'indicazione applicativa piu' ricorrente per alzare il recall e' generare piu' riformulazioni della
domanda con un modello generativo e fondere le liste. Qui non serve generarle: MTRAG fornisce tre
formulazioni per ogni turno — l'ultimo messaggio da solo, tutti i turni concatenati, e la riscrittura
autosufficiente — che da sole valgono molto diversamente (recall@10 con BGE: 0.356, 0.196, 0.417) ma
sbagliano su domande diverse.

| liste fuse | recall@10 | tetto a 50 | tetto a 100 | tetto a 200 | nDCG@10 |
|---|---|---|---|---|---|
| riscritta × 3 metodi (configurazione attuale) | 0.403 | 0.687 | 0.768 | 0.827 | 0.324 |
| riscritta + ultimo turno × 3 metodi | 0.418 | **0.700** | 0.779 | 0.845 | 0.331 |
| 3 formulazioni × 3 metodi | 0.425 | 0.679 | **0.781** | **0.848** | 0.328 |
| 3 formulazioni, solo BGE | 0.396 | 0.660 | 0.744 | 0.805 | 0.300 |
| 3 formulazioni, solo modelli densi | **0.437** | 0.688 | 0.763 | 0.835 | **0.340** |

Due risultati non ovvi. Fondere tre formulazioni usando un solo modello peggiora rispetto al punto di
partenza (0.396 contro 0.403): la varieta' delle domande non sostituisce la varieta' dei modelli,
serve che le liste sbaglino in modo diverso e due formulazioni lette dallo stesso encoder sbagliano
in modo simile. E BM25 si comporta in modo opposto alle due estremita' della lista — toglierlo
migliora le prime dieci posizioni (0.437 contro 0.425) e peggiora il tetto in profondita' (0.835
contro 0.848) — quindi la configurazione giusta dipende da cosa deve alimentare: se la lista e' il
risultato finale conviene senza BM25, se deve nutrire un reranker conviene con BM25, perche' li'
serve che l'oro sia presente, non che sia gia' in alto.

Il guadagno resta modesto in assoluto: due punti di tetto a profondita' 100 e 200. Non e' stato
eseguito un test appaiato su queste differenze, quindi vanno prese come indicative: su 180 domande,
scarti di questa dimensione sono al confine della risoluzione del campione.

## 10. Il sistema completo, e il fatto che le leve non si sommano

Montando insieme le due leve che avevano funzionato separatamente — lista costruita con due
formulazioni per tre metodi, poi riordino con il cross-encoder piccolo:

| configurazione | recall@10 | nDCG@10 |
|---|---|---|
| BGE-small da solo, com'era all'inizio del progetto | 0.417 | 0.332 |
| fusione multi-formulazione, senza riordino | 0.418 | 0.331 |
| formulazione singola + riordino sui primi 30 | 0.487 | 0.389 |
| **multi-formulazione + riordino sui primi 50** | **0.489** | **0.394** |

La multi-formulazione, sopra il reranking, aggiunge 0.002 di recall e 0.005 di nDCG: praticamente
nulla, nonostante da sola alzasse il tetto da 0.687 a 0.700. I due guadagni si sovrappongono invece
di sommarsi, perche' il reranking stava gia' estraendo quasi tutto l'oro che la lista conteneva, e
l'oro aggiunto dalla seconda formulazione e' in gran parte lo stesso che il riordino avrebbe
comunque trovato. L'unico effetto residuo e' lo spostamento della profondita' ottimale da 30 a 50,
coerente con il tetto piu' alto.

Merita attenzione anche la seconda riga: la fusione RRF di tre metodi, senza riordino, **non batte
BGE-small da solo** (0.418 contro 0.417). La fusione serve a costruire una lista di candidati con
piu' oro dentro — a profondita' 100 il vantaggio e' netto, 0.768 contro 0.726 — ma non a servire
direttamente i primi dieci risultati. E' l'esatto contrario di quanto scritto in Appunti6, dove la
fusione era stata adottata come configurazione di consegna.

Il bilancio del percorso: da 0.417 a 0.489 di recall@10 e da 0.332 a 0.394 di nDCG@10, circa il 18%
in piu' su entrambe, ottenuto con modelli gratuiti, senza ricalcolare gli embedding del corpus e
aggiungendo un modello da 22 milioni di parametri. Il tetto misurato con 200 candidati era pero'
0.83: il sistema ne converte meno dei due terzi, e le strategie di combinazione provate qui — piu'
modelli, piu' profondita', diversificazione, interpolazione, cascata — non colmano quel divario.

## 11. Decisioni di design e cosa resta davvero da provare

- **Configurazione adottata**: lista di candidati da fusione RRF (con BM25 dentro, che serve al
  tetto anche se danneggia le prime posizioni), riordino con `ms-marco-MiniLM-L-6-v2` sui primi
  30-50. Il reranking e' l'unica leva che ha prodotto un guadagno grande e riproducibile.
- **Scartate perche' misurate e inefficaci su questo corpus**: interpolazione con il punteggio di
  prima fase (α scelto su meta' campione risulta 1.0), diversificazione MMR, cascata fra due modelli
  di qualita' simile, fusione di reranker della stessa famiglia. Nessuna di queste e' sbagliata in
  generale: sono inefficaci *qui*, e il motivo e' sempre lo stesso, i modelli disponibili sbagliano
  nello stesso modo.
- **Da rifare con attenzione**: la valutazione usa 180 domande e un solo insieme di annotazioni
  incomplete. Le differenze sotto 0.02 osservate qui non vanno interpretate, e i confronti fra
  strategie vicine andrebbero fatti con un test appaiato, non guardando le medie.
- **La leva del modello di embedding, misurata: molto meno di quanto prometteva.** Incorporare tutto
  il corpus con `gte-base-en-v1.5` (6,4 ore di CPU, tre spegnimenti del portatile attraversati) ha
  prodotto il risultato piu' scomodo della sezione: **da solo il modello e' peggiore di bge-small**,
  0.300 contro 0.332 di nDCG@10, mentre su BEIR/FiQA e' dato 0.487 contro 0.403. Il banco pubblico non
  si e' trasferito, e ha invertito l'ordine: quelle domande sono dirette, queste vengono da
  conversazioni.

  | configurazione | recall@10 | nDCG@10 |
  |---|---|---|
  | bge-small da solo | 0.417 | 0.332 |
  | gte-base da solo | 0.375 | 0.300 |
  | candidati di bge-small riordinati da gte-base | 0.412 | 0.320 |
  | candidati di bge-small + cross-encoder | 0.478 | 0.387 |
  | **candidati di gte-base + cross-encoder** | **0.489** | **0.395** |

  Due conclusioni che valgono oltre questo progetto. La prima: **un modello va giudicato nel ruolo che
  ricoprira'**, non in astratto. gte-base peggiora il ranking diretto e insieme migliora la lista di
  candidati — il tetto a 50 sale da 0.687 a 0.711 — perche' recuperare e ordinare sono compiti
  diversi. La seconda: **un bi-encoder, per quanto forte, non sostituisce un cross-encoder nel
  riordino**. Usarlo per rileggere i 50 candidati scelti da un modello piu' debole da' 0.412, sotto
  il punto di partenza di 0.417 e molto sotto lo 0.478 di un cross-encoder venti volte piu' piccolo:
  il costo si paga per interrogazione in entrambi i casi, ma solo il secondo legge domanda e passaggio
  insieme.

  Il guadagno netto finale e' **+0.011 di recall@10** (da 0.478 a 0.489) per sei ore di calcolo: reale,
  riproducibile, e sproporzionato rispetto al costo. Con un servizio esterno lo stesso corpus si
  incorpora in minuti per 22 centesimi — e resterebbe da verificare se quel modello, su queste
  domande, si comporti meglio di quanto ha fatto gte-base.

- **Bilancio finale della sezione.** Tutte le leve disponibili sono state misurate: profondita' della
  lista, diversificazione, interpolazione dei punteggi, fusione di reranker, cascata, formulazione
  della domanda, multi-formulazione, reranker grande, modello di embedding piu' forte. Una sola ha
  prodotto un guadagno ampio — il riordino con cross-encoder, da 0.324 a 0.389 di nDCG@10 con un
  modello da 22 milioni di parametri — e tutte le altre rendono fra zero e 0.011. La configurazione
  finale vale 0.489 di recall@10 e 0.395 di nDCG@10, contro 0.417 e 0.332 del punto di partenza.
  Il tetto misurato a 200 candidati resta 0.82: il sistema ne converte meno dei due terzi, e nessuna
  combinazione provata colma quel divario su questo hardware.

## Coda: fondere reranker di famiglie diverse — l'unica leva rimasta che funziona

*Aggiunto il 19 settembre 2026, dopo la chiusura formale della sezione.*

Il bilancio qui sopra elencava la fusione di reranker fra le leve che non rendono nulla. Era vero per
quello che si poteva misurare allora, e non lo e' piu'.

`mxbai-rerank-xsmall` era stato dichiarato inutilizzabile nello step 11b: circa venti minuti per
domanda su questa CPU, contro un secondo e mezzo del cross-encoder MiniLM. Il calcolo pero' non era
stato interrotto, e' proseguito in sottofondo per giorni e si e' concluso: i punteggi di tutte le 180
domande esistono. Lo step 19 ripete lo step 12b senza toccarlo, con il terzo reranker incluso; lo
step 20 verifica la differenza con un confronto appaiato, che e' lo strumento giusto e finora mancante
per due metodi valutati sulle stesse domande.

| a parita' di lista e di profondita' (primi 30) | nDCG@10 | recall@10 |
|---|---|---|
| ms-marco-MiniLM-L6 da solo | 0.389 | 0.487 |
| mxbai-rerank-xsmall da solo | 0.393 | 0.484 |
| RRF dei due MiniLM (stessa famiglia) | 0.381 | 0.480 |
| **RRF MiniLM-L6 + mxbai (famiglie diverse)** | **0.417** | 0.501 |
| RRF di tutti e tre | 0.414 | 0.497 |

Il confronto appaiato contro MiniLM-L6 da solo, su 180 domande:

| confronto | differenza nDCG@10 | intervallo 95% | domande meglio / peggio | Wilcoxon |
|---|---|---|---|---|
| stessa famiglia | −0.008 | −0.017 … +0.001 | 35 / 45 | p = 0.15 |
| famiglie diverse | **+0.028** | **+0.009 … +0.048** | 68 / 38 | **p = 0.005** |
| mxbai da solo | +0.005 | −0.027 … +0.037 | 61 / 68 | p = 0.83 |

Tre cose si leggono insieme. **La diversita' conta piu' della qualita'**: mxbai da solo e' equivalente
a MiniLM-L6 — differenza 0.005, 61 domande migliorate contro 68 peggiorate, nessun segno di
superiorita' — eppure fonderlo con MiniLM-L6 produce l'unico guadagno statisticamente solido di tutta
la sezione dopo il reranking stesso. Fondere invece i due MiniLM, che vengono dalla stessa famiglia,
non porta niente e anzi peggiora di poco: due modelli che sbagliano nello stesso modo non aggiungono
informazione, e la famiglia architetturale e' un buon indicatore di come sbagliano. L'ipotesi era
scritta nello step 11b prima di poterla verificare, ed e' l'unica scommessa del progetto che si e'
rivelata giusta.

**Il guadagno e' sull'ordine, non sulla quantita'.** L'intervallo sul nDCG esclude lo zero, quello sul
recall no (+0.014, da −0.010 a +0.036, p = 0.17). La fusione riordina meglio gli stessi trenta
candidati, non ne porta di nuovi nei primi dieci: e' esattamente quello che ci si deve aspettare da un
metodo che opera su una lista gia' fissata, ed e' anche il motivo per cui il tetto di 0.82 resta
lontano.

**Il costo rende la leva inservibile qui**, e questo e' il punto che va nel banco accanto al numero:
+0.028 di nDCG@10 costano venti minuti per domanda invece di un secondo e mezzo, cioe' sessanta ore
per una valutazione completa. Con una GPU la stessa fusione costerebbe frazioni di secondo ed e'
la configurazione che si sceglierebbe in produzione; su questo portatile e' una misura, non un
sistema. La riga del banco recita percio' resa +0.028, costo ×850, e il vincolo «senza GPU» la
spegne — che e' la forma in cui una scelta di progetto va conservata.

Il bilancio della sezione va corretto di conseguenza: le leve che rendono qualcosa sono **due**, il
riordino con cross-encoder e la fusione di reranker di famiglie diverse, e la seconda e' fuori portata
per l'hardware disponibile.
