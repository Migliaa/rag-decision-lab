# Appunti 3 — Tappa 1: la prima misura, e perché ci ha ingannati

*Riscritta il 20 settembre 2026 con il senno di poi. La versione originale presentava i suoi numeri
come risultati; qui gli stessi numeri sono presentati per quello che si è scoperto essere — la misura
di un campione troppo piccolo — accanto ai valori corretti. Niente è stato tolto: il metodo di misura
spiegato qui è la parte che è rimasta valida e serve a leggere tutto il resto del percorso.*

**Posizione nel percorso**: è la tappa in cui si impara a misurare e si prende la prima decisione di
progetto sbagliata. Vedi [Appunti10](Appunti10.md) per il percorso completo.

Figure: **Fig3.1** = `output/metriche_12_domande.png` (script `06_metriche_12_domande.py`, pilota da
512 passaggi); **Fig3.2** = `output/metriche_corpus_completo.png` (script
`07_metriche_corpus_completo.py`, corpus FiQA completo, 61.022 passaggi — stesse 12 domande, cambia
solo la scala del corpus). Dati: `output/metriche_12_domande.json`,
`output/metriche_corpus_completo.json`.

## Cosa si stava cercando di decidere

Quale motore di ricerca usare come base: ricerca lessicale, ricerca densa, o entrambe. Per deciderlo
serviva saper misurare, quindi questa tappa introduce insieme la metrica e il primo confronto —
**BM25** (conta le parole in comune pesate per rarità, nessun embedding), **MiniLM** e **BGE-small**
(due bi-encoder). L'oro usato è quello esatto di ciascun turno dalle qrels.

## I due numeri che sono stati misurati, e il terzo che mancava

| Metodo | pilota, 512 passaggi (Fig3.1) | corpus completo, 12 domande (Fig3.2) | corpus completo, 180 domande (Appunti8) |
|---|---|---|---|
| BM25 | 0.653 | 0.056 | **0.211** |
| MiniLM | 0.958 | 0.264 | **0.369** |
| BGE-small | 1.000 | 0.167 | **0.417** |

*(recall@10 medio; la terza colonna è la misura valida, fatta molto più tardi)*

Le prime due colonne furono misurate qui, la terza no, e la differenza fra la seconda e la terza è
l'errore che ha condizionato tre tappe del progetto. Sulle 12 domande del pilota, a scala piena,
BGE-small risultava **peggiore** di MiniLM; sulle 180 è il migliore dei tre con un margine ampio.
L'ordine di merito non era sbagliato per poco, era invertito.

La spiegazione è banale e per questo istruttiva: 12 domande con in media tre passaggi oro ciascuna
sono 36 osservazioni binarie, e l'intervallo di confidenza su una media del genere vale circa ±0.20 —
più largo della differenza fra i metodi confrontati. Ogni conclusione tratta da quella tabella stava
dentro il rumore, e nulla nella tabella lo segnalava.

## Cosa fu deciso qui, e cosa ne è rimasto

Tre decisioni, prese guardando quei numeri:

1. **BGE-small come motore principale** — presa guardando il pilota, sbagliata come ragionamento
   (il pilota era troppo facile per distinguere i metodi) e **giusta per caso**: sulle 180 domande
   BGE-small è davvero il migliore dei tre. È rimasta, ma nessun merito va al modo in cui fu presa.
2. **BM25 non da solo, ma nemmeno scartato** — motivata da due casi letti a mano, non dalla media, ed
   è la parte migliore di questa tappa. Sulla domanda «resale value of my EV» BM25 ottiene nDCG@10 =
   1.000 contro 0.631 dei due densi, perché la domanda e il passaggio condividono i termini esatti;
   sulla domanda «what role do employees play in making good business decisions» BM25 ottiene
   recall@10 = 0.00 mentre entrambi i densi trovano tutto, perché lì serve il significato e non le
   parole. Due metodi che sbagliano in punti diversi sono il presupposto della fusione, e quel
   ragionamento ha retto: è diventato la tappa successiva.
3. **MiniLM da scartare, BGE lo batte quasi ovunque** — sbagliata nel merito e nel metodo. Sulle 180
   domande MiniLM sta sotto BGE ma la fusione dei tre sistemi resta il modo migliore di costruire la
   lista di candidati, quindi MiniLM è rimasto nel sistema finale. Scartare un componente perché ha
   una media più bassa, senza guardare se sbaglia sulle stesse domande, è l'errore di metodo da non
   ripetere.

## Il risultato scomodo che fu visto correttamente

Il passaggio dal pilota al corpus completo fece crollare tutti e tre i metodi — BGE da 1.000 a 0.167
di recall@10 — e quel crollo fu riconosciuto per quello che era invece di essere spiegato via. Prima
di accettarlo fu verificata l'integrità dei dati, perché gli embedding di BGE erano stati calcolati a
pezzi su tre sessioni interrotte da altrettanti spegnimenti del portatile: nessuna riga della matrice
a norma zero, e l'embedding di un passaggio noto (`448260-0-1360`) identico fra il calcolo del pilota
e quello del corpus completo, coseno 1.000000. Il crollo era reale.

La conclusione che ne fu tratta — **il pilota da 512 passaggi era troppo facile per essere usato come
prova di qualità** — è corretta e vale ancora. Quello che non fu visto è che la stessa obiezione si
applicava alle 12 domande, non solo ai 512 passaggi: fu corretta la scala del corpus e lasciata
intatta la scala del campione di valutazione, e il secondo errore è sopravvissuto alla correzione del
primo.

## Il metodo di misura, che è la parte durevole di questa nota

### Cos'è un passaggio

L'unità cercabile: non un documento intero ma un pezzo di documento già tagliato. Il documento
StackExchange `10171` è tagliato in almeno due passaggi, `10171-0-2129` (dal carattere 0 al 2129) e
`10171-1606-2596` (dal 1606 al 2596, quindi in parte sovrapposto al primo); il taglio l'ha fatto IBM
preparando MTRAG, non noi. Il corpus FiQA completo ne contiene 61.022. Il pilota da 512 era un
sottoinsieme di questi — i pochi passaggi oro per le 12 domande più molti riempitivi — costruito per
poter sperimentare in fretta su CPU.

### Perché il recall di una sola domanda può valere 0.33 senza ripetere nulla

Il denominatore non è il numero di prove ma il numero di passaggi oro di quella specifica domanda:

```
recall@10 (per QUESTA domanda) = (passaggi oro trovati nei primi 10) / (passaggi oro totali per questa domanda)
```

La domanda «What are the main factors to think about when buying a business?» ha 3 passaggi oro e
BM25 ne trova 1 nei primi 10, quindi 1/3 = 0.33. La domanda «Is it advisable to do day trading in an
IRA account» ne ha 4, BM25 ne trova 2, quindi 0.50. È un conteggio esatto su un'unica domanda.

Quando una domanda ha un solo passaggio oro il suo recall può valere soltanto 0 o 1, non perché la
metrica sia binaria ma perché con denominatore 1 non esistono altre frazioni; con più oro compaiono i
valori intermedi. La granularità dipende da quanti passaggi giusti ha quella domanda.

### Perché nDCG è quasi sempre frazionario anche con un solo oro

nDCG pesa anche **dove** l'oro è finito in classifica. Sulla domanda dell'auto elettrica, un solo oro,
BM25 ottiene nDCG@10 = 1.000: non solo l'ha trovato, l'ha messo al primo posto. Se lo stesso passaggio
fosse finito terzo il recall sarebbe rimasto 1.00 e l'nDCG sarebbe sceso a circa 0.5.

Il calcolo con i numeri veri di una domanda (MiniLM su «What is an IRA?», 3 oro alle posizioni 3, 5, 7):

```
DCG@10  = 1/log2(3+1) + 1/log2(5+1) + 1/log2(7+1) = 0.500 + 0.387 + 0.333 = 1.220
IDCG@10 = 1/log2(1+1) + 1/log2(2+1) + 1/log2(3+1) = 1.000 + 0.631 + 0.500 = 2.131
nDCG@10 = 1.220 / 2.131 = 0.573
```

Il numeratore usa le posizioni reali, il denominatore le stesse posizioni nella classifica ideale.

### Cosa rappresenta il «10»

Il numero di risultati che ci si permette di guardare, che è una scelta di progetto legata a un
vincolo a valle: quanti passaggi si passeranno al generatore. Se il budget di contesto ne accetta tre,
la metrica rilevante è recall@3, e uno stesso metodo può sembrare ottimo a k=10 e mediocre a k=3 se il
suo oro tende a stare fra il quarto e il decimo posto. La scelta di k va legata a come il recupero
verrà usato, non lasciata al valore predefinito di un tutorial. In M2 questo torna: il contesto del
generatore sono dieci passaggi, e la domanda giusta diventa quante *domande* ricevono almeno una fonte
utile, non quante fonti vengono recuperate in media.

### La barra e i punti in Fig3.1

La barra è la media aritmetica dei dodici valori, i punti neri sono i dodici valori singoli. Servono
insieme perché BM25 con media 0.653 oscilla fra 0.00 e 1.00 a seconda della domanda: non è
«moderatamente bravo» su ogni domanda, è perfetto su alcune e cieco su altre. Guardare la dispersione
accanto alla media è l'abitudine che avrebbe dovuto far sospettare del campione, ed è stata l'unica
avvisaglia disponibile a quel punto.

## Criteri di scelta del motore di ricerca, che restano validi

Formulati qui e mai smentiti, perché sono criteri e non misure:

- **Scala del corpus**: con milioni di passaggi il costo di mantenere un indice denso — spazio, tempo
  di re-indicizzazione a ogni aggiornamento — pesa molto di più, mentre un indice BM25 resta economico
  da aggiornare. La scelta fra denso puro e ibrido dipende anche da quanto spesso il corpus cambia.
- **Stile delle domande**: un pubblico che scrive codici prodotto, ticker e sigle sposta il vantaggio
  verso BM25; un pubblico che parafrasa lo sposta verso il denso. Va verificato sulle domande reali
  del proprio caso, non dedotto dal dominio.
- **Gergo specialistico**: se il corpus usa terminologia che un embedding generico ha visto poco in
  addestramento la qualità del denso cala, ed è l'argomento a favore di un embedding di dominio.
- **Latenza per domanda**: BM25 consulta un indice invertito e non deve incorporare la domanda con un
  modello, quindi costa meno per interrogazione.
- **Affidabilità della misura**: un insieme di valutazione piccolo indica una direzione, non una
  certezza. Scritto qui come avvertenza generica, e disatteso nella stessa nota.

## Le opzioni non misurate, e cosa ne è stato

Elencate qui come attese da verificare. Due sono poi state misurate e hanno smentito l'attesa:

- **BGE-base/large, E5, GTE**: attesa di un nDCG più alto a costo proporzionale, «più parametri
  aiutano». Misurato in Appunti8: `gte-base-en-v1.5`, dato 0.487 su BEIR/FiQA contro 0.403 di
  bge-small, su queste domande fa **0.300 contro 0.332**. L'attesa era sbagliata e il motivo è che i
  banchi pubblici usano domande diverse.
- **Nomic-embed-text**: nessun vantaggio atteso perché i passaggi sono già brevi. Mai misurato,
  l'attesa resta ragionevole.
- **Voyage-finance-2**: l'alternativa più promettente sulla carta per questo dominio, mai misurata per
  il vincolo di spesa. È diventata la prima riga del test a pagamento proposto in
  [Appunti11](Appunti11.md), dove si scopre che misurarla costerebbe 66 centesimi.
- **OpenAI/Cohere generiche**: attese a metà fra bge-small e le specializzate, mai misurate.
