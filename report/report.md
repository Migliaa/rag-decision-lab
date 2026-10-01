# Scegliere un sistema di ricerca con i dati

*Un percorso di studio su RAG, agenti e wiki compilate, in cui ogni scelta di progetto è stata
provata sul proprio caso d'uso prima di essere presa.*

---

## Sintesi

Un progetto di studio sul funzionamento dei sistemi di recupero e risposta (RAG) e delle loro
alternative, la ricerca guidata da un agente e la wiki compilata da un modello, per imparare ad
adattare il sistema al contesto in cui verrà usato. Ogni scelta è nata confrontando più soluzioni
plausibili sui dati delle prove, raccolti in uno strumento costruito apposta per decidere.

*(62 parole)*

---

## 1 · Il punto di partenza: domande, brani e distanze

Un sistema RAG risponde a una domanda usando documenti recuperati da un archivio invece della sola
memoria del modello. L'archivio viene diviso in brani, e ogni brano viene trasformato in un vettore
numerico in modo che testi di significato simile finiscano vicini; la domanda subisce la stessa
trasformazione, e il recupero consiste nel prendere i brani più vicini a lei e passarli al modello
che scrive la risposta.

**Figura**: `fig-1-domanda-fra-i-brani.png` — un campione di brani dell'archivio proiettato su un
piano, con una domanda e i dieci brani che il sistema le ha trovato vicini: tre sono quelli che il
benchmark indica come risposta, gli altri sette sono vicini per argomento ma non annotati.

Per misurare se il sistema trova i brani giusti serve un insieme di domande di cui si conoscono le
fonti. Il progetto usa la parte finanziaria di [MTRAG](https://github.com/IBM/mt-rag-benchmark),
un benchmark pubblico di conversazioni realistiche su un forum di finanza personale, con le fonti
annotate a mano per ogni domanda. Le annotazioni però sono incomplete: un brano recuperato e non
annotato non è necessariamente sbagliato, e questo limite ha pesato su ogni lettura dei risultati
successivi.

## 2 · Il metodo: uno strumento per decidere

Ogni componente di un sistema RAG ha più alternative ragionevoli — il modello che trasforma il
testo in vettori, il tipo di ricerca, il modello che riordina i risultati, il modello che genera
la risposta — e la scelta giusta dipende dal contesto: quanto si può spendere, su quale hardware
gira, quanto conta la velocità, se i dati possono uscire dall'azienda. Per non decidere a occhio
ho costruito un banco delle decisioni, una pagina in cui ogni componente ha la propria tabella di
alternative, con i criteri che contano per quella scelta affiancati al risultato, e i vincoli del
contesto attivabili come filtri che escludono le opzioni incompatibili.

**Figura**: `fig-0-banco.png` — il modulo del banco dedicato al riordino dei risultati, con i
valori misurati in questo progetto distinti da quelli dichiarati dai fornitori e dalle opzioni
mai provate.

La regola che regge lo strumento è che ogni valore dichiara da dove viene, misurato qui, dichiarato
da altri o non ancora rilevato, e solo i valori misurati possono essere evidenziati come la scelta
migliore. Non era così nella prima versione, dove un risultato preso da una classifica pubblica
compariva sopra tutte le opzioni provate davvero e avrebbe spinto verso un modello mai verificato
sul caso d'uso.

## 3 · Il recupero: scegliere misurando

Le prime misure, fatte per rapidità su una dozzina di domande, ordinavano i metodi in modo opposto
a quello che è emerso sull'intero insieme, e da lì ogni confronto è stato fatto su tutte le domande,
con un intervallo di incertezza e un test appaiato quando due opzioni erano vicine.

**Figura**: `fig-2-scelte-misurate.png` — a sinistra due modelli vettoriali secondo una classifica
pubblica e misurati su queste domande; a destra l'effetto di passare al riordinatore liste di
candidati sempre più lunghe. La qualità della lista misura quanto in alto compaiono le fonti giuste
fra i primi dieci risultati.

Il primo dei due risultati che hanno contraddetto le mie aspettative riguarda la scelta del
modello: la classifica pubblica misura un compito diverso, domande
isolate invece di conversazioni, e il vantaggio del modello più grande non si trasferisce. Il
secondo riguarda il riordino, che da solo vale quasi tutto il guadagno del progetto: dare al
riordinatore più candidati aumenta le fonti giuste disponibili, ma la quota che finisce nei primi
dieci cresce sempre meno, il massimo cade a cinquanta candidati senza distinguersi in modo
affidabile da trenta, e oltre comincia a scendere mentre il tempo per domanda continua a salire.
A parità di risultato ho tenuto trenta, che costa meno. La combinazione di due riordinatori di
famiglie diverse ha dato l'unico guadagno ulteriore confermato dal test appaiato, ma a un costo di
venti minuti per domanda sul computer usato: nel banco resta registrata come il margine disponibile,
non come la scelta.

## 4 · La generazione: cambia l'unità di misura

Il recupero si giudica brano per brano, la risposta domanda per domanda, e le due misure non
coincidono: a chi legge la risposta interessa solo se nel contesto c'era abbastanza per rispondere.

**Figura**: `fig-3-dove-finisce-la-fonte.png` — per ogni domanda, dove finisce la fonte giusta
migliore: in cima alla lista, comunque nel contesto passato al modello, trovata ma esclusa dal
contesto, oppure mai trovata.

Leggere a mano un caso per ciascuna di queste situazioni ha mostrato un modo di fallire che nessuna
metrica di recupero vede: un contesto pertinente per argomento ma fatto di frammenti di discussione
che non contengono una risposta. Il modello che genera è stato scelto per costo, non per qualità:
provare e confrontare più modelli locali avrebbe richiesto giorni di lavoro manuale per imparare
poco, mentre un modello a pagamento via API costava pochi dollari per tutte le prove, e nel banco
le alternative locali restano segnate come non misurate.

Il confronto più utile è stato isolare le domande in cui la configurazione scelta e quella di
partenza passano al modello un contesto con un numero diverso di fonti giuste: in circa un terzo di
quei casi la configurazione migliore in media porta meno fonti, e generando le risposte con lo
stesso modello la differenza non è mai fra una risposta giusta e una sbagliata, ma nella
completezza. Scegliere fra due sistemi guardando solo se la media sale significa accettare, senza
saperlo, che una parte delle domande peggiori.

## 5 · Agenti e wiki: chi guida la ricerca

In un RAG la ricerca è fissata dal codice, una volta per domanda e sempre nello stesso modo; in un
agente è il modello a decidere, passo dopo passo, cosa cercare, cosa leggere e quando fermarsi,
secondo lo schema descritto da Yao et al. nel 2022 ([ReAct](https://arxiv.org/abs/2210.03629)).
La differenza non sta in dove si cerca, archivio locale, pagine compilate o web, ma in chi controlla
il ciclo della ricerca.

**Figura**: `fig-4-agente-e-wiki.svg` — le due architetture a confronto con il RAG: quello che ciascuna lascia al modello e quello che resta al codice.

Ho costruito un agente minimo, con due strumenti per cercare e leggere i brani e un limite di sei
passi imposto dal codice, e l'ho messo alla prova su due domande dello stesso benchmark. Sulla
prima, una domanda che il recupero fisso non riusciva a risolvere, lo stesso modello che davanti a
un contesto insufficiente si era astenuto correttamente ha speso il budget cercando varianti della
domanda invece di leggere, e avvicinandosi al limite ha dato una risposta sicura e infondata: la
pressione di un tetto di passi spinge a produrre qualcosa, un rischio che un sistema a passo unico
non corre. Sulla seconda, scelta perché la fonte giusta si raggiunge solo con due ricerche in
sequenza, l'agente è rimasto bloccato oltre venti minuti su una singola chiamata al modello, perché
il codice limitava il numero di passi ma non la durata di ciascuno. Un agente ha bisogno di due
tetti imposti dall'esterno, uno sui passi e uno sul tempo, e con questa progettazione non è ancora
un sistema da mettere davanti a chi fa una domanda.

La wiki compilata sposta il lavoro di collegare le fonti da ogni domanda a una sintesi preparata in
anticipo, come nel [GraphRAG](https://arxiv.org/abs/2404.16130) di Microsoft, e il suo meccanismo
di ricerca è quello dell'agente applicato a pagine scritte da un modello. Non l'ho costruita: la
convenienza dipende dal rapporto fra quante domande arrivano e quanto spesso cambiano le fonti,
perché ogni cambiamento obbliga a ricompilare le pagine che ne dipendono, e su un archivio di prova
senza traffico quel rapporto non si può misurare. Nel banco la scelta resta segnata come ragionata,
non misurata.

Molti dei limiti incontrati dipendono dai modelli di oggi e si ridurranno, ma le domande con cui
sono stati trovati restano le stesse: da dove viene un numero, se il miglioramento medio vale per
ogni caso, quali limiti il sistema deve avere a prescindere da quanto il modello sia bravo a
fermarsi da solo.
