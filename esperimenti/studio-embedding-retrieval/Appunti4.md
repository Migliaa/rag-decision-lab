# Appunti 4 — RRF, architetture RAG, agenti e wiki (teoria, per studiare offline)

Nota diversa dalle precedenti: qui non c'è una figura né un risultato misurato in questo progetto
— è materiale di preparazione teorica, a livello professionale, da leggere prima di costruire e
misurare queste parti. Ogni termine nuovo rispetto a quanto
già visto in Appunti1-3 è definito al primo uso. Indice: `INDICE_APPUNTI.md`.

---

## 1. Combinare più sistemi di ricerca: Reciprocal Rank Fusion (RRF)

### 1.1 Il problema che RRF risolve

In Appunti3 abbiamo misurato BM25, MiniLM e BGE separatamente sulle stesse domande: ciascuno
sbaglia in punti diversi (BM25 vince quando la domanda condivide parole esatte col passaggio
giusto, perde quando serve capire il significato). Il passo successivo naturale è **non scegliere
un solo metodo, ma combinare le loro classifiche** in una sola.

Il problema tecnico: i punteggi non sono confrontabili tra loro. Un punteggio BM25 è una somma di
pesi di rarità delle parole — non ha un massimo fisso, dipende dal corpus. Un punteggio di
similarità coseno tra due embedding normalizzati è sempre tra -1 e 1. Sommare o mediare
direttamente questi due numeri non ha senso: un punteggio BM25 di 8.3 non è "più alto" o "più
basso" di un coseno di 0.4 in nessun modo comparabile — sono unità di misura diverse. **Servirebbe
prima normalizzare entrambi su una scala comune** (es. min-max: riscalare i punteggi di ogni
classifica in modo che il più alto diventi 1 e il più basso 0) per poterli sommare in modo
sensato — un approccio possibile, chiamato **CombSUM** quando si sommano i punteggi normalizzati di
più sistemi, **CombMNZ** quando si moltiplica quella somma per il numero di sistemi in cui il
documento compare (premia chi è stato recuperato da più metodi, non solo con punteggio alto in
uno solo).

**RRF prende una strada diversa: ignora i punteggi, usa solo la POSIZIONE in classifica.** Ogni
sistema di retrieval produce un ordinamento (1°, 2°, 3°... posto); RRF combina gli ordinamenti,
non i numeri che li hanno prodotti. Vantaggio: funziona sempre, indipendentemente da come è fatto
il punteggio di partenza (BM25, coseno, o qualunque altra cosa), senza dover scegliere una
normalizzazione arbitraria.

### 1.2 La formula, passo per passo

Per un passaggio che compare in una o più classifiche:

```
RRF_score(passaggio) = Σ  1 / (c + rank_sistema(passaggio))
                       sistemi in cui compare
```

- `rank_sistema(passaggio)` è la posizione di quel passaggio in QUELLA classifica (1, 2, 3...). Se
  il passaggio non compare affatto nella classifica di un sistema (es. non è nei primi 1000 di
  BM25), quel sistema semplicemente non contribuisce alla somma per quel passaggio.
- `c` è una costante piccola, quasi sempre **60** nella pratica (il valore usato nel paper
  originale del 2009 e diventato uno standard de facto, non derivato da una teoria — un caso raro
  in cui un "magic number" è semplicemente convenzione condivisa dal settore). Serve ad ammorbidire
  la differenza tra le prime posizioni: senza `c`, il 1° posto (1/1 = 1.0) peserebbe enormemente di
  più del 2° (1/2 = 0.5); con c=60, 1° posto vale 1/61 ≈ 0.0164 e il 2° vale 1/62 ≈ 0.0161 — la
  differenza tra 1° e 2° diventa piccola, quella tra 1° e 50° resta comunque grande.

**Esempio numerico costruito a mano**, due sistemi (BM25 e BGE) su 3 passaggi (A, B, C), c=60:

| Passaggio | Rank in BM25 | Rank in BGE | RRF score |
|---|---|---|---|
| A | 1 | 3 | 1/61 + 1/63 = 0.01639 + 0.01587 = **0.03226** |
| B | 2 | 1 | 1/62 + 1/61 = 0.01613 + 0.01639 = **0.03252** |
| C | 3 | 2 | 1/63 + 1/62 = 0.01587 + 0.01613 = **0.03200** |

Classifica finale RRF: B (0.03252) > A (0.03226) > C (0.03200). B vince pur non essendo mai al 1°
posto assoluto in nessuno dei due sistemi, perché è costantemente vicino alla cima in entrambi —
esattamente l'effetto voluto: premiare chi è "abbastanza buono ovunque" rispetto a chi è
"eccezionale in un solo sistema, mediocre nell'altro".

### 1.3 Estensioni pratiche

- **Più di due sistemi**: la formula si estende sommando un termine per ogni sistema aggiuntivo
  (es. BM25 + MiniLM + BGE, tre termini invece di due). Nessun limite teorico al numero di
  sistemi da fondere.
- **RRF pesato**: si può assegnare un peso diverso a ciascun sistema,
  `Σ w_sistema / (c + rank_sistema(passaggio))`, se si ha ragione di credere che un sistema sia
  più affidabile di un altro sul proprio dominio (es. dare più peso a BGE se le misure mostrano
  che è sistematicamente più forte, ma senza azzerare il contributo di BM25). Il peso è un
  iperparametro da scegliere/misurare, non una formula.
- **Limite di RRF**: buttando via l'informazione di punteggio, si perde la differenza tra "1°
  posto con margine enorme sul 2°" e "1° posto per un pelo" — RRF li tratta allo stesso modo.
  Quando i punteggi grezzi sono affidabili e comparabili (stesso tipo di sistema, es. due embedding
  diversi ma entrambi con coseno normalizzato), una media pesata dei punteggi può conservare più
  informazione di RRF. RRF è la scelta robusta di default quando i sistemi sono eterogenei (come
  BM25 + un embedding), non necessariamente la scelta con più informazione in assoluto.
- **Alternative più sofisticate**: **learning to rank** — invece di una formula fissa (RRF,
  CombSUM...), si addestra un piccolo modello (spesso un semplice regressore o un gradient
  boosting) che impara da esempi etichettati come combinare i segnali di più sistemi (posizioni,
  punteggi, lunghezza del passaggio, numero di parole in comune...) nel modo che massimizza
  nDCG sul proprio set di validazione. Richiede dati etichettati (qrels) in quantità superiore a
  quella di un pilota da 12 domande — un investimento che si giustifica solo a una scala
  produttiva reale, non in fase di prototipo.

### 1.4 Dove si inserisce il reranking

RRF fonde le classifiche di più **retriever di primo stadio** (BM25, bi-encoder come MiniLM/BGE —
veloci, applicabili a tutto il corpus). Il passo successivo, se serve ancora più precisione in
cima, è il **reranking con cross-encoder**:

- Un **bi-encoder** (quello usato finora: MiniLM, BGE) trasforma query e passaggio SEPARATAMENTE
  in due vettori indipendenti, poi li confronta con un prodotto scalare. Per questo si può
  pre-calcolare la matrice del corpus una volta sola: il passaggio non "sa" nulla della query
  mentre viene incorporato.
- Un **cross-encoder** riceve query e passaggio **concatenati insieme** come un unico input, e il
  modello produce direttamente un punteggio di pertinenza per quella coppia specifica — può
  cogliere interazioni fini tra le parole delle due (es. una negazione nella query che cambia
  completamente la pertinenza di un passaggio altrimenti simile), cosa che un bi-encoder, guardando
  i due testi separatamente, può perdere. Il costo: va ricalcolato da zero per OGNI coppia
  query-passaggio, quindi è troppo lento per scorrere un intero corpus di migliaia/milioni di
  passaggi ad ogni domanda.
- **Schema completo in produzione**: BM25 + bi-encoder(i) → fusione RRF → primi 20-100 candidati
  → cross-encoder rilegge SOLO quei candidati → riordino finale → primi k (es. 10) passati al
  generatore. Prodotti reali usati per questo passo: Cohere Rerank, BGE-reranker-large (stessa
  famiglia BAAI usata per l'embedding), modelli `cross-encoder/ms-marco-*` di sentence-transformers,
  Voyage rerank-2. Quanti candidati rileggere (20? 50? 100?) è un compromesso costo/qualità da
  misurare: più candidati, più probabilità di recuperare un oro che il primo stadio aveva messo
  in basso, ma più tempo di calcolo per domanda.

---

## 2. Dal recupero alla risposta: l'architettura di un RAG

### 2.1 Termini nuovi, definiti prima di usarli

- **Generatore (o modello generativo)**: il modello linguistico (LLM, Large Language Model) che
  riceve testo e produce testo in risposta — a differenza dei modelli di embedding, che
  trasformano testo in vettori e basta, il generatore scrive una risposta in linguaggio naturale.
- **Prompt**: il testo effettivamente inviato al generatore, costruito da noi — non solo la
  domanda dell'utente, ma un pacchetto che include istruzioni, i passaggi recuperati, e la domanda.
- **Finestra di contesto (context window)**: il numero massimo di token (unità di testo, non
  esattamente parole) che il generatore può ricevere in un prompt in una volta sola. È un limite
  fisico del modello, non una scelta di design — ma quanto DI QUELLA finestra riempire con
  passaggi recuperati sì che lo è.
- **Citazione/attribuzione**: far scrivere al generatore da quale passaggio viene ogni affermazione
  (es. "[fonte: 448260-0-1360]"), per poter verificare dopo se la risposta è davvero supportata.

### 2.2 La pipeline completa

```
domanda utente
   → retrieval (BM25 + embedding → RRF → eventuale rerank, sezione 1)
   → assemblaggio del contesto (i top-k passaggi scelti, con i loro ID)
   → costruzione del prompt (istruzioni + contesto + domanda)
   → chiamata al generatore (LLM)
   → risposta, idealmente con citazioni verso i passaggi usati
```

### 2.3 Come si costruisce il contesto: dettagli che contano in pratica

- **Ordine dei passaggi nel prompt non è neutro.** È documentato (fenomeno noto in letteratura
  come "lost in the middle", perso nel mezzo) che i modelli linguistici tendono ad attenzionare
  meglio l'informazione all'inizio e alla fine di un contesto lungo, peggio quella nel mezzo — a
  parità di quanto sia effettivamente rilevante. Conseguenza pratica: il passaggio più rilevante
  (secondo il punteggio di retrieval) va tipicamente messo per primo o per ultimo nel prompt, non
  sepolto a metà di dieci passaggi.
- **Il budget di passaggi (il "k" di Appunti3) è vincolato dalla finestra di contesto**, non
  scelto in astratto: se il generatore accetta 8.000 token e ogni passaggio ne occupa in media 300,
  il budget realistico è nell'ordine delle decine di passaggi, non centinaia — motivo per cui
  Recall@k va misurato con il k che si intende davvero usare in produzione, come già notato in
  Appunti3.
- **Il prompt di sistema (system prompt)** istruisce esplicitamente il generatore a rispondere
  SOLO sulla base dei passaggi forniti, e a dichiarare esplicitamente quando l'informazione non è
  presente ("non è specificato nelle fonti fornite") invece di completare con conoscenza propria.
  Riduce le allucinazioni ma non le elimina: un generatore può comunque ignorare l'istruzione o
  mescolare fonte e conoscenza pregressa in modo non distinguibile dall'esterno.

### 2.4 L'effetto degli errori a monte (retrieval) sulla risposta generata

Due modi distinti in cui il retrieval può sbagliare, con conseguenze diverse sulla risposta:

**Errore di recall — il passaggio giusto non è stato trovato affatto.** Il generatore non ha
accesso all'informazione corretta, per costruzione. Due esiti possibili: (a) con un prompt
disciplinato, il generatore dichiara di non sapere — corretto anche se insoddisfacente; (b) senza
questa disciplina, il generatore "riempie il vuoto" con la propria conoscenza pregressa (spesso
plausibile, a volte sbagliata, e senza modo per l'utente di distinguere le due cose) — questa è
un'**allucinazione indotta dal retrieval**, non un difetto del generatore in sé.

**Errore di precisione — passaggi irrilevanti finiscono nel contesto insieme a quelli giusti (o al
posto loro).** È documentato che i generatori possono "distrarsi": la presenza di testo irrilevante
nel contesto peggiora la qualità della risposta anche quando il passaggio giusto è comunque
presente, non solo quando manca. Non è un fallimento garantito — un buon generatore ignora spesso
il rumore — ma non è un'ipotesi su cui affidarsi senza misurarla.

**Come si diagnostica quale dei due sta succedendo:** è precisamente il motivo per cui si misurano
SEPARATAMENTE le metriche di retrieval (recall@k, nDCG@k, viste in Appunti2/3) e le metriche sulla
risposta finale generata (faithfulness, context precision — RAGAS, già introdotto in Appunti2). Se
il recall misurato è basso, il problema è a monte: si interviene sul chunking, sull'embedding,
sulla fusione ibrida — non sul prompt. Se il recall è alto ma la risposta è comunque scorretta o
non supportata, il problema è a valle: prompt, generatore, o mancanza di un controllo post-hoc.
Confondere questi due livelli porta a "aggiustare" la parte sbagliata del sistema.

**Come si gestisce in pratica:**

- **Lato retrieval**: aumentare la qualità a monte (ibrido, reranking — sezione 1) piuttosto che
  sperare che il generatore compensi un cattivo recupero.
- **Lato generazione**: istruzioni esplicite a citare le fonti e a rifiutarsi quando l'evidenza è
  insufficiente; un **controllo post-hoc** — una seconda chiamata al modello (o un modello più
  piccolo dedicato) che verifica se ogni affermazione della risposta è davvero supportata dal
  contesto fornito, bloccando o segnalando la risposta se non lo è. È letteralmente il modo in cui
  si misura la faithfulness di RAGAS, applicato in produzione invece che solo in valutazione
  offline.
- **Lato architettura**: una **soglia di confidenza** — se il punteggio del miglior passaggio
  recuperato è sotto una soglia scelta (segno che probabilmente nessuna fonte buona esiste nel
  corpus per questa domanda), il sistema evita del tutto la generazione e risponde "nessuna fonte
  pertinente trovata", invece di forzare comunque una risposta con fonti deboli.

---

## 3. Ricerca tramite agenti (agentic retrieval)

### 3.1 Cosa cambia rispetto al RAG fisso descritto sopra

Tutto il paragrafo 2 descrive un **RAG a un solo passaggio (single-shot)**: una domanda, un
retrieval, una generazione, fine. Un **agente**, in questo contesto specifico (non un'intelligenza
artificiale generica autonoma — un termine spesso usato in modo vago), è un generatore a cui viene
dato il retrieval come uno **strumento (tool)** che può invocare più volte, decidendo da solo
quante volte cercare, con quali domande, e quando fermarsi. Il generatore stesso guida il processo
di ricerca invece di riceverlo già confezionato.

### 3.2 Pattern concreti, con terminologia professionale

- **Decomposizione della domanda (query decomposition)**: una domanda complessa a più parti
  ("confronta il rendimento di due fondi e dimmi quale ha commissioni più basse") viene spezzata
  dal modello in sotto-domande più semplici, ciascuna cercata separatamente, poi le risposte
  parziali vengono combinate.
- **ReAct (Reason + Act)**: il pattern più citato in letteratura per gli agenti basati su LLM — il
  modello alterna esplicitamente un passo di **ragionamento** (testo che spiega cosa fare e
  perché, non mostrato all'utente finale) e un passo di **azione** (es. "cerca: rendimento fondo
  X 2024"), osserva il risultato dell'azione, e decide il prossimo ragionamento — un ciclo che
  continua finché il modello decide di avere abbastanza informazione per rispondere.
- **Ricerca iterativa / follow-up**: dopo una prima ricerca, il modello può accorgersi che manca
  un pezzo (es. ha trovato il rendimento del fondo ma non le commissioni) e lanciare una seconda
  ricerca mirata, invece di rispondere con l'informazione parziale.
- **Uso di strumenti multipli**: la ricerca nei documenti è spesso solo UNO degli strumenti
  disponibili all'agente, insieme ad altri (calcolatrice, interrogazione di un database SQL,
  ricerca web) — il modello decide quale usare in base alla domanda.
- **Riflessione/autocritica (self-critique)**: dopo aver scritto una bozza di risposta, un passo
  aggiuntivo verifica se è ben supportata (simile al controllo post-hoc della sezione 2.4);
  se non lo è, l'agente può decidere di cercare ancora invece di consegnare la risposta.

### 3.3 Quando conviene, quando no

**Vantaggi**: gestisce domande multi-hop (che richiedono di combinare informazioni da fonti
diverse in più passaggi), domande di conversazione dove serve capire cosa manca rispetto al
contesto precedente, e casi dove una singola ricerca fissa dimostrabilmente non basta (misurabile:
se il recall@k di un retrieval fisso su domande complesse è sistematicamente basso, è un indizio a
favore di un approccio agentico che possa riformulare e ricercare più volte).

**Costi e rischi, non ipotetici**: ogni passo di ragionamento/azione è una chiamata separata al
modello — latenza e costo (per token) si moltiplicano per il numero di passi, spesso 3-10 volte
quello di un RAG a passaggio singolo. Serve un **limite esplicito di passi** (altrimenti il ciclo
ricerca→ragiona→ricerca può non fermarsi mai su domande ambigue). È più difficile da valutare e
da debuggare rispetto a una pipeline fissa: l'errore può nascere in una qualunque delle iterazioni,
non in un unico punto misurabile con recall@k.

**Strumenti/framework professionali reali** (non tutti usabili gratis in locale, citati perché
rilevanti per il mondo del lavoro, coerente con quanto già chiarito in Appunti1): **LangGraph**
(orchestrazione di agenti a grafo, open source), **LlamaIndex** (agenti e RAG, open source con
opzioni a pagamento per servizi gestiti), il ciclo agentico nativo di Claude tramite tool use
(chiamata di funzioni/strumenti durante la conversazione, quello su cui è basato anche questo
stesso assistente).

---

## 4. Wiki costruita da un LLM (wiki compilation)

### 4.1 L'idea, in contrasto con il RAG

Il RAG (sezioni 2-3) ricerca e rilegge i documenti grezzi **ad ogni domanda**, anche se mille
utenti fanno domande molto simili tra loro. Una **wiki compilata da LLM** inverte l'ordine: un
processo offline (non a runtime, non mentre l'utente aspetta una risposta) usa un generatore per
leggere le fonti UNA VOLTA e produrre pagine strutturate, riassunte, organizzate per argomento — un
lavoro di compilazione pagato una sola volta (o periodicamente), dopo il quale le domande future
consultano direttamente le pagine già pronte, senza dover rileggere e ri-sintetizzare i documenti
grezzi ogni volta.

### 4.2 Decisioni di design specifiche

- **Provenienza (provenance)**: ogni affermazione nella pagina wiki deve poter essere ricondotta al
  passaggio/documento sorgente da cui è stata sintetizzata — stesso principio delle citazioni nel
  RAG (sezione 2.3), ma applicato al momento della compilazione invece che ad ogni risposta.
- **Aggiornamento (staleness)**: quando una fonte cambia, la pagina wiki compilata da essa diventa
  potenzialmente obsoleta — serve una politica esplicita di quando ricompilare (equivalente al
  problema di invalidazione di una cache, una delle aree già previste nel progetto). Senza una
  politica esplicita, la wiki può servire informazione vecchia con la stessa sicurezza apparente di
  una aggiornata, perché l'utente non vede la differenza tra una pagina fresca e una obsoleta.
- **Il rischio specifico di questo approccio**: un errore di sintesi commesso dal LLM in fase di
  compilazione (un'allucinazione durante la scrittura della pagina, non durante una risposta
  isolata) diventa **persistente**: viene servito a tutte le domande future su quell'argomento
  finché qualcuno non lo scopre e ricompila la pagina. È peggio, in un certo senso, di
  un'allucinazione in un RAG a passaggio singolo (sezione 2.4): lì l'errore capita una volta per
  quella domanda; qui si "cristallizza" e si ripete silenziosamente.

### 4.3 Quando conviene

Conviene quando lo stesso nucleo di fatti viene interrogato ripetutamente da molti utenti (il
costo di compilazione si ammortizza sul volume di domande future) — non conviene per domande rare
o molto specifiche ("long tail"), dove il costo di compilare una pagina dedicata supera quello di
una semplice ricerca RAG occasionale su quell'argomento.

---

## 5. Tabella riassuntiva — le tre strade dello stesso confronto (D4 del progetto)

| Approccio | Lavoro per domanda | Lavoro anticipato/periodico | Quando conviene |
|---|---|---|---|
| RAG fisso (sezione 2) | Un retrieval + una generazione | Nessuno oltre l'indicizzazione | Domande semplici, singolo salto informativo, bassa tolleranza a latenza/costo per domanda |
| Agente sui documenti grezzi (sezione 3) | Più cicli ricerca-ragionamento, nessun lavoro anticipato oltre l'indicizzazione | Nessuno | Domande complesse/multi-hop, conversazioni con contesto da ricostruire, quando il RAG fisso dimostrabilmente manca informazione |
| Agente su wiki compilata (sezione 4) | Lettura di pagine già pronte, spesso più veloce del RAG grezzo | Compilazione iniziale + ricompilazione quando le fonti cambiano | Stesso nucleo di argomenti interrogato spesso da molti utenti, tolleranza a un costo di manutenzione continuo |

Nessuna delle tre è "la migliore" in assoluto: la scelta segue lo stesso principio di tutto questo
studio — motivarla con il compito, la qualità misurata e le risorse disponibili, non con la moda
del momento.
