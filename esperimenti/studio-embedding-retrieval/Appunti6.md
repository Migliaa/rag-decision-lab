# Appunti 6 — Tappa 2: fondere le classifiche, e scegliere quanto in profondità guardare

*Riscritta il 20 settembre 2026 con il senno di poi. I numeri restano quelli misurati allora, sulle
12 domande del pilota, affiancati ai valori corretti sulle 180. Il ragionamento di progetto è
riportato intero, perché è la parte che ha retto: due delle tre decisioni prese qui sono ancora nel
sistema finale.*

**Posizione nel percorso**: è la tappa in cui si smette di scegliere un componente e si comincia a
comporne diversi, e in cui compare per la prima volta la distinzione fra il tetto del sistema e
quanto ne converte. Vedi [Appunti10](Appunti10.md).

Figura: **Fig6.1** = `output/fusione_rrf.png` (script `08_fusione_rrf.py`, corpus FiQA completo, 61.022
passaggi; gli embedding del corpus erano già calcolati, qui si incorporano solo le 12 domande).
Dati: `output/fusione_rrf.json`.

## Il problema da cui si parte

La tappa precedente aveva mostrato che nessuno dei tre motori, da solo, arriva a un livello
utilizzabile sul corpus completo. Restavano due domande: la fusione delle tre classifiche recupera
qualcosa, e quanto deve essere lunga la lista di candidati da passare poi al reranking.

## Come fu affrontato, che è la parte da conservare

Due decisioni di metodo, prese prima di eseguire:

1. **Tre schemi di pesi in parallelo invece di uno scelto a tavolino.** La proposta iniziale dava il
   peso più alto a BGE, ma la misura appena fatta mostrava BGE peggiore di MiniLM a scala piena —
   conclusione che si sarebbe poi rivelata falsa, ma il riflesso fu giusto: invece di fidarsi
   dell'ipotesi si confrontarono lo schema proposto, il suo inverso, e uno uniforme come riferimento
   neutro. Confrontare più configurazioni costa secondi quando gli embedding sono già calcolati, e
   questa abitudine ha poi retto per tutto il progetto.
2. **La profondità del reranking si misura, non si assume.** Invece di stimare a tavolino quanti
   candidati servissero, fu misurato sulla classifica fusa a quale posizione arriva davvero ogni
   passaggio oro. È il primo uso della nozione che diventerà centrale: la **copertura**, cioè quanto
   oro è presente nella lista, che è indipendente da quanto il riordino poi ne porta in cima.

## Risultato di allora, e la misura valida

| k guardato | proposto | invertito | uniforme (12 domande) | **uniforme (180 domande)** |
|---|---|---|---|---|
| 10 | 0.243 | 0.312 | 0.264 | **0.403** |
| 50 | 0.556 | 0.500 | 0.479 | **0.687** |
| 100 | 0.715 | 0.771 | 0.812 | **0.768** |
| 500 | 1.000 | 1.000 | 1.000 | — |

Profondità dell'oro nella lista fusa, sulle 12 domande e i tre schemi insieme: mediana 86, novantesimo
percentile 415, massimo 496. Sulle 180 domande la distribuzione è molto più compatta, ed è la ragione
per cui la decisione presa qui sulla lunghezza della lista si è poi rivelata sbagliata.

Cambia anche una conclusione qualitativa: a k=10 la fusione **non** batte bge-small da solo (0.403
contro 0.417), mentre a k=100 lo supera nettamente (0.768 contro 0.726). La fusione serve a costruire
la lista di candidati, non a servire direttamente i primi dieci risultati — cosa che le 12 domande
lasciavano intravedere e che le 180 rendono netta.

## Osservazioni di allora, con il loro destino

- **Nessuno schema di pesi vince a tutti i k**, e con 12 domande differenze di quella entità non
  bastano a dichiarare un vincitore. L'osservazione era corretta e fu scritta, ma non portò alla
  conseguenza ovvia — allargare il campione — bensì a scegliere lo schema uniforme come compromesso.
  Riconoscere che i dati non sostengono una scelta e poi scegliere comunque è stato l'errore
  ricorrente di questa fase del progetto.
- **La fusione aiuta solo a k grandi.** Vero allora e vero sulle 180: è esattamente il ruolo che la
  fusione ha nel sistema finale.
- **Il novantesimo percentile della profondità dell'oro è 415**, quindi fissare la lista a 100
  candidati significa perdere in partenza l'oro delle domande peggiori. L'osservazione era onesta e
  la conseguenza pratica fu opposta a quella giusta: si scelse di guardare **più** in profondità
  possibile entro il budget, mentre la misura sulle 180 domande ha poi mostrato che rerankare 30
  candidati batte rerankarne 100 o 200, perché ogni candidato in più è anche un'occasione in più che
  il riordinatore sbagli. La copertura sale con la profondità, la conversione scende più in fretta.

## Le tre decisioni prese qui

1. **Schema di pesi uniforme (33/33/33)**, scelto perché i dati non sostenevano nessuna ipotesi di
   superiorità. Sopravvissuta: sulle 180 domande la differenza fra gli schemi resta piccola e lo
   schema uniforme è nel sistema finale.
2. **Fusione come costruttore della lista di candidati, non come classifica finale.** Sopravvissuta e
   rafforzata.
3. **Lista di 100 candidati per il reranking.** Rovesciata in Appunti8: la profondità migliore è 30,
   e a 200 candidati il risultato peggiora rispetto a 30 nonostante la copertura sia molto più alta
   (0.827 contro 0.618). È la decisione più istruttiva di questa tappa perché fu presa con il criterio
   giusto — misurare la copertura — applicato a metà: fu misurato il tetto e non la conversione, cioè
   quanta parte del tetto il sistema riesce davvero a portare nei primi dieci.

## Cosa è rimasto aperto, e come è finita

La nota chiudeva indicando il chunking ereditato da IBM come probabile causa a monte del tetto.
L'ipotesi fu verificata in Appunti8 con un conteggio diretto: i 61.022 passaggi provengono da 57.638
documenti distinti, cioè 1,06 passaggi per documento, quindi nella quasi totalità dei casi il
«passaggio» è il documento intero e non c'è alcun taglio da incolpare. Il chunking è stato scagionato,
e l'ipotesi — ragionevole, scritta con chiarezza, e falsa — è costata due tappe di sospetto verso il
posto sbagliato.
