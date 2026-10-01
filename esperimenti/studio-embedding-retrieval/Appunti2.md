# Appunti 2 — Misurare una classifica: recall, nDCG e il resto

Figura di riferimento: **Fig2.1** = `output/query_vicini.png` (script `04_figura_query_vicini.py`).
Script di calcolo: `03_query.py` (classifica), `05_metriche_singola_query.py` (metriche). Indice:
`INDICE_APPUNTI.md`.

Nota di lavoro dello studio libero, stesso status di `Appunti1.md`: confluirà nel report di modulo
quando la consolidazione lo richiederà.

## Cosa mostra Fig2.1

Domanda: "What is an IRA?" (turno 3 di una conversazione del pilota MTRAG/FiQA, oro noto dalle
qrels: 3 passaggi). La figura posiziona la query nella stessa proiezione t-SNE del corpus,
rifatta da zero includendo la query (t-SNE non supporta l'inserimento di un punto in una mappa già
calcolata, a differenza di PCA — vedi commento in testa allo script). Verde = passaggio che il
modello ha messo in top-10 E che le qrels segnano rilevante. Cerchio arancione = in top-10 ma le
qrels non lo segnano rilevante. Linee tratteggiate = distanza dalla query a ciascun passaggio oro,
per rendere visibile se la vicinanza nel disegno corrisponde a un vero risultato trovato.

Un'osservazione onesta, non nascosta: nella mappa MiniLM il risultato con il **punteggio più alto
in assoluto** (`108391-0-361`, il 1° in classifica reale) finisce disegnato lontano dal gruppo
query+oro. È il limite della proiezione già discusso in Appunti1: la vicinanza nel disegno 2D non
è affidabile quanto il punteggio calcolato sui 384 numeri originali. La figura lo dimostra invece
di limitarsi ad affermarlo.

## Il processo di valutazione, passo per passo

1. Si parte da una domanda con **oro noto**: un insieme di passaggi che una persona (non il
   sistema) ha giudicato rilevanti per quella domanda. Senza questo, non si può misurare niente —
   si può solo guardare un punteggio e credere che sia buono.
2. Il sistema produce una **classifica** di passaggi ordinati per punteggio, indipendentemente
   dall'oro (il modello non sa quali sono i passaggi giusti mentre calcola i punteggi).
3. Si confrontano i primi *k* della classifica con l'oro noto. Da qui in poi, metriche diverse
   rispondono a domande diverse sullo stesso confronto.

## Recall@k e nDCG@k sulla domanda di Fig2.1

**Recall@k** — quota di oro che compare da qualche parte nei primi *k* risultati, senza guardare
la posizione esatta:

```
recall@k = (numero di passaggi oro trovati nei primi k) / (numero totale di passaggi oro)
```

**nDCG@k** (normalized Discounted Cumulative Gain) — premia l'oro trovato in alto nella classifica
più di quello trovato in fondo:

```
DCG@k  = somma, per ogni oro trovato in posizione r ≤ k, di 1 / log2(r + 1)
IDCG@k = lo stesso calcolo nella classifica IDEALE (tutto l'oro compattato ai primi posti)
nDCG@k = DCG@k / IDCG@k
```

La divisione per IDCG@k serve a rendere il numero confrontabile tra domande con quantità diverse
di oro: una domanda con 5 passaggi rilevanti ha più margine di punteggio grezzo di una con 2, a
prescindere da quanto bene lavora il retriever — normalizzare toglie questo effetto.

Risultato calcolato da `05_metriche_singola_query.py` su questa domanda:

| Modello | Posizioni dell'oro in classifica | Recall@10 | nDCG@10 |
|---|---|---|---|
| MiniLM | 3, 5, 7 | 1.000 | 0.573 |
| BGE-small | 1, 2, 6 | 1.000 | 0.933 |

Il recall è identico (entrambi trovano tutto l'oro nei primi 10): da solo non distingue i due
modelli. L'nDCG sì — BGE mette l'oro molto più in alto (1°, 2°, 6° posto contro 3°, 5°, 7°), e
questo si traduce in un nDCG quasi doppio. È la ragione per cui in pratica non ci si ferma mai a
una sola metrica: il recall dice "il sistema ha trovato tutto?", l'nDCG dice "lo ha messo dove
conta?" — due domande diverse sulla stessa classifica, e qui danno risposte diverse. Resta un
singolo esempio: non generalizza da solo (prossimo passo: le 12 domande del pilota).

## Altre metriche usate in ambito professionale

| Metrica | Cosa misura | Quando si preferisce |
|---|---|---|
| **Precision@k** | Quota dei primi *k* risultati che sono effettivamente rilevanti (l'opposto speculare del recall, che guarda quanto oro è coperto) | Quando il costo di un risultato sbagliato in cima conta — es. i primi *k* passaggi finiscono nel prompt del generatore in un RAG, e un passaggio irrilevante spreca spazio e può confondere la risposta |
| **MRR** (Mean Reciprocal Rank) | 1 / posizione del PRIMO risultato rilevante, mediato su più domande | Quando conta solo trovare una risposta buona il prima possibile, non tutte — tipico delle ricerche "voglio una sola fonte affidabile", non di una rassegna completa |
| **MAP** (Mean Average Precision) | Media della precision calcolata a ogni posizione in cui compare un risultato rilevante, poi mediata su più domande | Quando serve un singolo numero riassuntivo che tenga conto sia di precisione che di posizione, usato spesso nei benchmark IR storici (TREC) |
| **Hit Rate@k** | Percentuale di domande che hanno ALMENO un risultato rilevante nei primi *k* (binario per domanda, non per passaggio) | Nei sistemi conversazionali dove basta agganciare l'argomento giusto, il dettaglio lo recupera un passo successivo |

### Metriche specifiche per un RAG completo (oltre al solo retrieval)

Le metriche sopra giudicano la classifica di passaggi, non la risposta finale generata. Per quella
si usano framework dedicati come **RAGAS**, con metriche giudicate spesso da un altro modello
linguistico (LLM-as-judge) invece che da un umano, perché generare giudizi umani su ogni risposta
non scala:

- **Faithfulness** — quota di affermazioni nella risposta effettivamente supportate dai passaggi
  recuperati (rileva le allucinazioni rispetto alle fonti).
- **Answer relevancy** — se la risposta affronta davvero la domanda posta, indipendentemente dal
  fatto che sia supportata dalle fonti.
- **Context precision / context recall** — le versioni di precision/recall applicate al contesto
  effettivamente inviato al generatore, non alla classifica grezza del retriever.

### Segnali in produzione, non offline

Le metriche sopra richiedono un oro noto (qrels) o un giudice (umano o LLM) preparato in anticipo.
In produzione, dopo il rilascio, si aggiungono segnali impliciti raccolti dagli utenti reali: click
sul risultato, tempo speso a leggerlo, pollice su/giù, tasso di riformulazione della domanda (segno
che la prima risposta non bastava). Sono più economici da raccogliere su grande scala ma più
rumorosi e indiretti — un click non prova che il contenuto fosse corretto, solo che sembrava
promettente.

## Scelte di design: quale metrica usare, o combinare

Non esiste "la" metrica giusta in assoluto — la scelta dipende da cosa deve fare il sistema:

- **Serve un'unica fonte affidabile** (es. assistente che cita un solo documento) → MRR o
  Precision@1 contano più del recall complessivo.
- **Serve una rassegna completa** (es. due diligence finanziaria, non si può permettere di perdere
  un documento rilevante) → il recall@k (con k abbastanza alto) è la metrica primaria.
- **Il contesto inviato al generatore ha un budget di token limitato** → precision@k pesa quanto o
  più del recall, perché ogni passaggio irrilevante incluso è spazio tolto a uno buono.
- **Serve un solo numero per confrontare configurazioni in automatico** (es. scegliere tra dieci
  combinazioni di chunking) → nDCG@k è lo standard più diffuso, perché in un solo numero tiene
  conto sia di cosa è stato trovato sia di dove.

In pratica professionale raramente si riporta una sola metrica: si riportano **insieme** recall@k
(la copertura è sufficiente?), nDCG@k o MRR (la posizione è buona?), e — se il sistema genera
risposte, non solo classifiche — le metriche di RAG complete sopra (la risposta è fedele alle
fonti?). Ogni metrica cattura un modo diverso di sbagliare; una sola non basta a diagnosticare
dove intervenire quando qualcosa non funziona.
