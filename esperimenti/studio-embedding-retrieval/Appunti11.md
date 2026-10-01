# Appunti 11 — Il test a pagamento: progetto, costo misurato, decisioni da prendere

Nota di progetto, non di risultato: descrive un esperimento non ancora eseguito. Serve a rispondere
alla domanda che M1 lascia aperta — *sappiamo davvero come si chiude il divario fra 0.489 e il tetto
di 0.82, e quanto costa?* — trasformandola da affermazione in misura.

La decisione se eseguirlo resta di Andrea. Quello che segue è il costo reale, il progetto
dell'esperimento, e le decisioni che il banco deve saper sostenere perché l'esperimento abbia senso.

## Il numero che cambia la cornice

Il corpus è stato misurato, non stimato: 61.022 passaggi, 45.807.309 caratteri, **11,45 milioni di
token**. Su quella base i listini danno questo:

| voce | modello | prezzo di listino | costo per questo corpus |
|---|---|---|---|
| incorporare il corpus | OpenAI text-embedding-3-small | 0,02 $ / M token | **0,23 $** |
| incorporare il corpus | Voyage voyage-finance-2 | ~0,06 $ / M token | **0,66 $** |
| incorporare il corpus | OpenAI text-embedding-3-large | 0,13 $ / M token | **1,49 $** |
| incorporare il corpus | Cohere embed-v4 | ~0,12 $ / M token | **1,37 $** |
| riordinare 180 domande × 30 candidati | Cohere Rerank 3.5 | 2 $ / 1000 ricerche | **0,36 $** |
| riordinare 180 domande × 30 candidati | Voyage rerank-2.5 | 0,05 $ / M token | **0,05 $** |

Incorporare le 180 domande costa frazioni di centesimo e non compare in tabella.

**Un confronto completo a pagamento su questo progetto costa circa 1,30 $.** Il vincolo che ha
governato M1 non era il denaro: era il tempo di CPU. Ventitré ore per incorporare il corpus con
`bge-large`, sei ore e mezza con `gte-base`, sessanta ore per una valutazione completa con
`mxbai-rerank-xsmall` — le stesse operazioni, comprate, costano fra venti centesimi e un euro e
mezzo e finiscono in minuti. È il risultato più scomodo di tutta la sezione, e va scritto così
com'è: la scelta di restare sul gratuito è stata pagata in giorni di calcolo, non in euro risparmiati.

Questo **non** significa che la decisione fosse sbagliata, e il motivo è di scala. A questo corpus il
costo è trascurabile; con sei milioni di passaggi invece di sessantunomila lo stesso lavoro costa
circa 23 $ con il modello piccolo e 149 $ con quello grande, e va rifatto a ogni cambio di modello e
a ogni aggiornamento del corpus. Il riordino peggiora: a centomila interrogazioni al mese, Cohere
Rerank costa 200 $ al mese, Voyage rerank-2.5 circa 28 $. **Il costo di un test non dice nulla sul
costo di un sistema**, e questa è la distinzione che il banco oggi non rappresenta bene — vedi sotto.

Nota sui listini: sono valori di settembre 2026 e vanno riverificati prima di spendere, perché
cambiano spesso. Nel banco sono marcati come *dichiarati*, non come misurati, che è esattamente il
motivo per cui quella colonna esiste.

## La via a costo zero, da verificare per prima

Prima di spendere qualunque cifra: Voyage e Jina offrono un credito iniziale in token gratuiti e
Cohere una chiave di prova a velocità ridotta, tutti sufficienti a coprire un corpus di questa
dimensione. Se i piani sono ancora quelli, **l'intero esperimento si esegue a costo zero**, pagando
in velocità invece che in denaro. Va verificato come primo passo, non assunto: sono le condizioni
commerciali che cambiano più spesso di tutto il resto.

## Il progetto dell'esperimento

Struttura in due stadi, che ricalca la distinzione imparata in M1 fra costruire la lista di candidati
e riordinarla, e che consente di attribuire ogni guadagno allo stadio che l'ha prodotto.

**Stadio A — la lista di candidati.** Si sostituisce il recupero di prima fase e si misura la
**copertura** a 30, 50 e 100 candidati, non il nDCG: è il criterio con cui in M1 si giudica questa
decisione, perché a questo stadio conta quanto oro entra nella lista e non come è ordinato. Due
opzioni, scelte per rispondere a due domande diverse:

- `text-embedding-3-small` (0,23 $) risponde a «quanto rende un modello generico di buona qualità»;
- `voyage-finance-2` (0,66 $) risponde a «quanto vale un modello addestrato su questo dominio», che è
  l'ipotesi più promettente sulla carta e l'unica che M1 non ha mai potuto toccare.

Il riferimento da battere è la copertura misurata: 0.618 a 30, 0.687 a 50, 0.768 a 100, e 0.827 a
200. La domanda decisiva non è se la copertura salga ma **di quanto**, perché il tetto a 200
candidati è 0.82 e il sistema ne converte meno dei due terzi: se la copertura sale poco, il divario
non è nel recupero di prima fase e il test lo dimostra spendendo un euro.

**Stadio B — il riordino.** Sui primi 30 della lista migliore, `Cohere Rerank 3.5` (0,36 $) e
`voyage rerank-2.5` (0,05 $), confrontati contro il migliore gratuito con lo stesso confronto
appaiato dello step 20 — differenza domanda per domanda, intervallo appaiato, test dei ranghi,
domande migliorate contro peggiorate. Le medie da sole non bastano a distinguere un guadagno da un
rumore, ed è l'errore che M1 ha commesso per tre tappe.

**Una conseguenza della struttura dei listini che vale come decisione di progetto.** Cohere fattura
a ricerca, con un massimo di cento documenti inclusi: riordinare 30 candidati o 100 costa
identicamente 0,36 $. Voyage fattura a token: passare da 30 a 100 candidati triplica il costo. Con
Cohere la profondità diventa un parametro gratuito e va esplorata tutta; con Voyage è una voce di
spesa. La stessa domanda tecnica ha due risposte diverse a seconda di come il fornitore fattura, ed è
il genere di cosa che non compare in nessun banco pubblico.

**Costo totale del progetto così com'è: circa 1,30 $**, che con i tentativi andati storti e una
seconda passata sulla configurazione migliore si arrotonda a sotto i 5 $.

## Cosa dimostrerebbe, e cosa no

Dimostrerebbe due cose, entrambe utili al portfolio più di un punteggio: che il divario noto è stato
**quantificato** invece che dichiarato, e che la scelta di restare sul gratuito è stata presa
conoscendo l'alternativa e il suo prezzo. Il caso in cui i modelli a pagamento rendessero poco è
altrettanto interessante del caso opposto, perché direbbe che il limite è nelle annotazioni e nella
natura conversazionale delle domande, non negli strumenti — ed è l'ipotesi che i risultati di M1
rendono plausibile.

Non dimostrerebbe che il sistema è pronto per la produzione: il campione resta di 180 domande con
annotazioni incomplete, e nessun costo di listino cattura la parte cara di un sistema reale, che è
tenerlo in piedi mentre il corpus cambia.

## Le decisioni da prendere, e dove il banco le sostiene

| decisione | modulo del banco | criterio | il banco basta? |
|---|---|---|---|
| quale embedding comprare | Modello di embedding | costo di indicizzazione, finestra, dominio, riproducibilità | **ora sì**, dopo la modifica qui sotto |
| quale riordinatore comprare | Reranker | costo per provarlo, costo a regime, latenza | **ora sì** |
| a quale profondità riordinare | Profondità dei candidati | copertura contro conversione | sì, già misurato |
| cosa misurare | Metrica di valutazione | cosa la metrica non vede | sì |
| se mandare i dati fuori | vincolo «dati riservati» | — | **no**, vedi sotto |

## Le modifiche al banco, fatte e da fare

Progettare questo test ha mostrato tre punti in cui il banco non sosteneva la decisione. Due sono
stati corretti, il terzo è una proposta.

**Fatto — il costo di indicizzazione era una stringa.** Il modulo embedding aveva una sola colonna di
testo che conteneva indifferentemente «~5 h CPU» e «0,22 $ · minuti»: due valute diverse nella stessa
casella, impossibili da confrontare e da ordinare. Ora sono due colonne, **ore di indicizzazione** e
**$ di indicizzazione**, e la riga che prima sembrava incomparabile diventa leggibile a colpo
d'occhio — 23 ore e zero dollari contro mezz'ora e 1,49 $.

**Fatto — mancava il costo per provare un'opzione.** Il modulo riordinatore aveva la spesa a regime
(`$ / 1k query`) ma non il costo di una singola valutazione su questo progetto, che è il numero con
cui si decide *cosa misurare dopo*. La colonna **`$ per provarlo`** trasforma il banco da registro di
ciò che si sa in piano di ciò che conviene rilevare: le righe con evidenza «non rilevato» e un costo
di prova basso sono la lista della spesa, in ordine di prezzo.

**Fatto — mancava la riproducibilità.** Un modello locale si blocca a una revisione esatta e resta
identico per sempre; un modello servito può cambiare o essere ritirato sotto i piedi, e i risultati
misurati smettono di essere riproducibili. In M1 è successa la versione speculare del problema — la
serie 5 di `transformers` ha rotto `gte-base` e ha richiesto di fissare la 4.57.6 — quindi il criterio
non è teorico. Colonna **`versione bloccabile`**, sì o no.

**Da fare — il vincolo «dati riservati» è troppo grossolano.** È un interruttore che spegne tutte le
opzioni esterne insieme, mentre nel lavoro la domanda è più fine: il fornitore addestra sui dati
inviati, esiste un'opzione di conservazione zero, dove risiedono fisicamente i dati, che certificazioni
ha. Un progetto con dati sanitari e uno con documentazione pubblica hanno entrambi «dati esterni»
acceso e decidono in modo opposto. Serve una colonna di testo per la politica di conservazione, e il
vincolo va sdoppiato fra «nessun dato esce» e «dati fuori con conservazione zero».

**Da fare — manca il volume, che è ciò che rende il costo una decisione.** Tutte le colonne di spesa
sono per mille interrogazioni, il che è corretto ma statico: la differenza fra 0,36 $ per un test e
200 $ al mese per un sistema sta interamente nel numero di interrogazioni, e quel numero non è da
nessuna parte nel banco. Un campo nell'intestazione del progetto — *interrogazioni al mese attese* —
con le colonne di spesa ricalcolate su quel valore renderebbe visibile il punto in cui un'opzione
economica smette di esserlo. È la modifica più utile fra quelle rimaste, e la sola che cambia la
struttura invece che le colonne.

**Da fare — l'unità di fatturazione.** Cohere fattura a ricerca, Voyage a token, e da questo dipende
se la profondità di riordino sia un parametro gratuito o una voce di costo. Una colonna di testo con
l'unità di fatturazione rende quella conseguenza leggibile senza doverla ricavare ogni volta.
