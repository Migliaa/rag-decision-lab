# Appunti 7 — Tappa 3: il reranking, e il primo limite strutturale riconosciuto

*Riscritta il 20 settembre 2026 con il senno di poi. I valori vengono dalle 12 domande del pilota e
sono sottostimati di circa due volte; sono conservati perché il ragionamento che ne fu tratto è
rimasto valido quasi per intero, ed è il ragionamento che ha determinato la forma del sistema finale.*

**Posizione nel percorso**: è la tappa in cui si distingue per la prima volta ciò che il sistema
potrebbe raggiungere da ciò che raggiunge, e in cui si vede che una media che migliora può nascondere
casi che peggiorano. Vedi [Appunti10](Appunti10.md).

Figura: **Fig7.1** = `output/reranking_bge_large.png` (script `09_reranking.py`, corpus completo).
Dati: `output/reranking_bge_large.json`.

## Cosa fa questa tappa

Esegue le due decisioni della tappa precedente — pesi uniformi, lista di 100 candidati — e aggiunge
il riordino con un cross-encoder, `BAAI/bge-reranker-large`.

## Bi-encoder e cross-encoder, che è il concetto portante di tutto M1

La differenza non è di scala ma di meccanismo. Un **bi-encoder** trasforma domanda e passaggio in due
vettori calcolati separatamente e li confronta dopo con un prodotto scalare, ed è per questo che gli
embedding del corpus si calcolano una volta e si riusano per sempre. Un **cross-encoder** prende
domanda e passaggio insieme, in un solo passaggio della rete, e legge le due parti l'una in funzione
dell'altra fin dall'inizio: coglie corrispondenze che il confronto fra due vettori indipendenti perde,
una condizione numerica che compare solo in un passaggio, una negazione. Il prezzo è che non esiste
una rappresentazione riusabile del passaggio, quindi ogni coppia richiede un passaggio completo della
rete e il metodo si applica solo a pochi candidati.

Questa distinzione è stata poi verificata sperimentalmente nel modo più diretto possibile (Appunti8):
usare un bi-encoder forte, `gte-base`, per riordinare i candidati scelti da un modello più debole dà
0.412 di recall@10, sotto il punto di partenza di 0.417 e molto sotto lo 0.478 di un cross-encoder
venti volte più piccolo. Il costo si paga per interrogazione in entrambi i casi, ma solo il secondo
legge la coppia insieme.

## Risultato di allora

| k | fusione sola | fusione + reranking |
|---|---|---|
| 1 | 0.028 | 0.090 |
| 10 | 0.264 | 0.319 |
| 50 | 0.479 | 0.736 |
| 100 | 0.812 | 0.812 |
| nDCG@10 | 0.166 | 0.250 |

Sulle 180 domande gli stessi ordini di grandezza diventano 0.403 → 0.478 di recall@10 e 0.324 → 0.387
di nDCG@10, con un cross-encoder da 22 milioni di parametri al posto di quello da 560.

## Il tetto, che è la nozione nata qui

Il recall@100 è identico prima e dopo il riordino non per coincidenza ma per costruzione: il reranker
riordina i cento candidati che riceve e non ne aggiunge. Se un passaggio oro non è nella lista,
nessun riordino successivo può farlo comparire. Il massimo raggiungibile resta fissato dalla fase di
recupero, e il riordino può solo spostare verso l'alto ciò che sta già dentro quel massimo.

Da qui viene la coppia di grandezze con cui è stato giudicato tutto il resto del progetto: il **tetto**
(quanto oro è presente nella lista) e la **conversione** (quale frazione di quel tetto finisce nei
primi dieci). Il sistema finale ha tetto 0.82 a duecento candidati e ne converte meno dei due terzi, e
la ragione per cui rerankare trenta candidati batte rerankarne duecento è che la conversione scende
più in fretta di quanto salga il tetto.

## Il guadagno medio non è un miglioramento uniforme

Sulle 12 domande il recall@10 migliorava in 5, restava invariato in 5 e **peggiorava in 2**:

| domanda | oro | rank nella fusione | rank dopo il riordino | recall@10 |
|---|---|---|---|---|
| «What advantage does Germany gain from repeatedly bailing out other countries?» | 2 | 9, 83 | 1, 2 | 0.50 → 1.00 |
| «What is an IRA?» | 3 | 72, 119, 351 | 10, 119, 351 | 0.00 → 0.33 |
| «Is it advisable to do day trading in an IRA account with a $5500 deposit?» | 4 | 21, 31, 67, 211 | 2, 36, 60, 211 | 0.00 → 0.25 |
| «Do the SEC rules on pattern day trading apply to IRAs?» | 3 | 1, 3, 26 | 1, 6, 8 | 0.67 → 1.00 |
| «What makes a good manager?» | 4 | 2, 9, 29, 57 | 1, 7, 10, 32 | 0.50 → 0.75 |
| «What are the main factors to think about when buying a business?» | 3 | 10, 52, 232 | 26, 73, 232 | 0.33 → 0.00 |
| «Should I consider investing in existing startups instead of starting my own?» | 3 | 6, 10, 13 | 12, 13, 32 | 0.67 → 0.00 |

I due peggioramenti non erano un errore nello script — stessi cento candidati, solo riordinati — ma il
cross-encoder che sbaglia su quelle due domande. La proporzione «2 su 12» non è affidabile, il
fenomeno sì: sulle 180 domande la fusione fra reranker di famiglie diverse, che è il guadagno più
solido della sezione, migliora 68 domande e ne peggiora 38.

Contare le domande migliorate accanto a quelle peggiorate è nato qui ed è diventato il modo standard
di presentare un guadagno in questo progetto, perché una differenza media può venire da un
miglioramento diffuso o da pochi casi estremi, e sono due cose che portano a decisioni diverse. Quello
che mancava ancora era lo strumento per dire se la differenza fosse rumore: il confronto appaiato è
arrivato solo in coda al progetto, nello step 20.

## Le decisioni di questa tappa

- **Il reranking non risolve il problema a monte.** Vero allora e vero adesso: il tetto è fissato dal
  recupero, e il riordino è lo strumento per scegliere bene fra i candidati presenti, non per
  rimediare a quelli assenti.
- **Il costo del reranker è delimitato per costruzione** — cento coppie per domanda invece dell'intero
  corpus — a differenza degli embedding. Questo lo rendeva l'unico punto del sistema dove un modello
  molto più grande sembrava accessibile, e fu la ragione per cui si scelse `bge-reranker-large`. La
  misura poi ha mostrato che su questa CPU costa 2,7 minuti per domanda, cioè otto ore per una
  valutazione completa, e che il modello da 22 milioni di parametri rende quasi quanto lui: il grande
  vale +0.006 per venti volte il costo. «Delimitato per costruzione» non significa praticabile.
- **Il chunking come causa a monte** restava indicato come sospetto principale. Scagionato in
  Appunti8 con un conteggio diretto: 1,06 passaggi per documento, quindi non c'è quasi nessun taglio.
