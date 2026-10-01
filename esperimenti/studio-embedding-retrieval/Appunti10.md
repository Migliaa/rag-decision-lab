# Appunti 10 — Il percorso decisionale di M1

Documento di raccordo: le altre note sono tappe, questa è il filo. Serve a leggere M1 come un
processo di decisioni prese, riviste e in tre casi rovesciate, invece che come un elenco di misure.
È il prodotto della sezione — non il punteggio finale, e nemmeno il banco decisioni, che è lo
strumento.

## La struttura del percorso

| tappa | nota | cosa si decideva | cosa fu deciso | destino |
|---|---|---|---|---|
| 1 | [Appunti3](Appunti3.md) | quale motore di ricerca | bge-small principale, BM25 come rete di sicurezza, MiniLM scartato | due su tre corrette, prese con un metodo sbagliato |
| 2 | [Appunti6](Appunti6.md) | come fondere, quanto guardare in profondità | pesi uniformi, fusione per i candidati, lista da 100 | le prime due reggono, la terza è rovesciata |
| 3 | [Appunti7](Appunti7.md) | se e come riordinare | cross-encoder grande sui 100 candidati | il principio regge, il modello no |
| 4 | [Appunti8](Appunti8.md) | tutto da capo, su 180 domande | dodici misure che sostituiscono le precedenti | è la misura valida del progetto |
| 5 | [Appunti9](Appunti9.md) | come rendere le scelte confrontabili | il banco decisioni | strumento, non risultato |
| 6 | coda di [Appunti8](Appunti8.md) | verifica appaiata delle differenze piccole | fusione fra famiglie diverse, misurata e non adottabile | l'ultimo risultato della sezione |

## Come si è evoluto il modo di decidere

La parte trasferibile del percorso non sono le scelte ma il criterio con cui sono state prese, che è
cambiato quattro volte.

**All'inizio si decideva sulla media di un confronto.** Tre metodi, tre numeri, si prende il più alto.
Bastò passare dal pilota al corpus completo perché la classifica si ribaltasse, e bastò passare da 12
domande a 180 perché si ribaltasse di nuovo in direzione opposta. Il difetto non era la metrica ma
l'assenza di una misura della sua incertezza: 12 domande con tre passaggi oro ciascuna danno un
intervallo di confidenza di circa ±0.20, più largo di tutte le differenze che si stavano
confrontando, e niente nella tabella lo diceva.

**Poi si è cominciato a confrontare configurazioni invece di modelli.** Nella tappa 2, invece di
scegliere uno schema di pesi a tavolino, ne furono misurati tre in parallelo — possibile perché gli
embedding del corpus erano già calcolati e il confronto costava secondi. Da lì in avanti ogni
esperimento ha separato **il calcolo costoso** (incorporare il corpus, calcolare i punteggi di un
riordinatore) **dalla strategia economica** (come usarli), salvando i primi su disco: è la ragione per
cui negli step 11 e 12 sono state confrontate una dozzina di strategie di riordino al prezzo di un
solo calcolo, e per cui quando `mxbai-rerank-xsmall` ha finito i suoi punteggi dopo giorni è bastato
rieseguire un confronto di secondi per usarli.

**Poi si è imparato a separare il tetto dalla conversione.** Nella tappa 3, osservando che il recall
alla profondità della lista non cambia dopo il riordino, è nata la coppia di grandezze con cui è stato
giudicato tutto il resto: quanto oro è presente nella lista, e quale frazione di quell'oro finisce nei
primi dieci. Senza quella distinzione la decisione sulla lunghezza della lista sembra ovvia — più
candidati, più copertura, meglio è — e infatti fu presa così e si rivelò sbagliata: a 200 candidati la
copertura è 0.827 e a 30 è 0.618, ma il risultato finale a 30 è migliore, perché ogni candidato in più
è anche un'occasione in più che il riordinatore sbagli.

**Infine si è imparato a chiedersi se una differenza sia reale.** L'ultimo passo della sezione (step
20) confronta due metodi domanda per domanda invece che media contro media, con un intervallo
appaiato e un test dei ranghi: due metodi valutati sulle stesse domande sbagliano insieme, quindi la
differenza appaiata ha una variabilità molto minore della differenza fra le due medie, e intervalli di
confidenza che si sovrappongono non significano che la differenza sia nulla. È lo strumento che
mancava alla tappa 1 e che avrebbe evitato tutto il resto.

## I tre errori, e cosa li ha resi possibili

**Il campione da 12 domande.** Non fu una svista: il limite era scritto nero su bianco in ognuna delle
tre note come avvertenza, e ogni volta la nota proseguiva a trarre conclusioni come se non lo fosse.
Riconoscere un limite e poi decidere comunque è il modo più efficiente di sbagliare, perché produce
documenti che sembrano prudenti. La correzione è arrivata solo quando il limite è stato trattato come
un compito da eseguire — rifare tutto su 180 domande — invece che come una nota a piè di pagina.

**Il chunking sospettato per due tappe.** L'ipotesi che i tagli ereditati da IBM limitassero il
recupero era ragionevole, scritta con chiarezza, e falsa. Verificarla richiedeva un conteggio di tre
righe — 61.022 passaggi da 57.638 documenti, cioè 1,06 passaggi per documento, quindi quasi nessun
taglio — che non fu fatto per due tappe mentre si costruiva sopra quel sospetto.

**Il modello più grande.** L'aspettativa che più parametri diano risultati migliori ha guidato due
scelte costose, il riordinatore grande e l'embedding forte, e in entrambi i casi la misura l'ha
smentita: `bge-reranker-large` rende +0.006 per venti volte il costo di un modello da 22 milioni di
parametri, e `gte-base`, dato 0.487 su BEIR/FiQA contro 0.403 di bge-small, su queste domande fa
0.300 contro 0.332. La lezione che ne è rimasta è che un modello va giudicato nel ruolo che occuperà e
sulle domande che riceverà, non nella classifica generale.

## Dove entra il banco decisioni, e dove non entra

Il banco è nato alla tappa 5, quando quasi tutte le decisioni erano già prese: non è lo strumento che
le ha prodotte, e scriverlo diversamente sarebbe una ricostruzione comoda. È lo strumento che le ha
rese **rivedibili**, e la differenza si è vista subito — messe nella stessa tabella, con costo e
guadagno accanto e l'origine di ogni numero dichiarata, tre righe sono risultate indifendibili e sono
state cambiate.

Quello che il banco ha effettivamente prodotto in M1:

- **Ha reso visibile che il criterio di scelta cambia da decisione a decisione.** La prima versione
  usava una griglia uniforme con nDCG@10 su ogni riga, e fu buttata: l'unità di indicizzazione non si
  sceglie guardando il nDCG, che dipende da tutta la pipeline a valle, ma su quante unità produce
  l'indice, quanti token contiene ciascuna, quante risposte restano spezzate e se la
  ri-segmentazione invalida le annotazioni. Costruire una tabella per ogni decisione obbliga a dire
  prima *su cosa* quella decisione si prende, che è la parte del lavoro che di solito resta implicita.
- **Ha separato ciò che è stato misurato da ciò che è dichiarato.** Ogni riga porta la propria
  origine: misurato qui, dichiarato da un fornitore o da un banco pubblico, non rilevato. Senza quella
  colonna il valore preso dalla scheda di un prodotto e quello ottenuto con una misura propria
  sembrano la stessa cosa, e la tappa su `gte-base` ha mostrato quanto costa confonderli.
- **Ha trasformato i vincoli nel meccanismo di scelta.** La domanda non è quale opzione sia migliore
  in assoluto ma quale resti praticabile date le condizioni: attivando insieme «dati riservati» e
  «budget zero», delle quattordici righe del modulo riordinatore ne restano otto e spariscono tutte le
  migliori per latenza. È la forma in cui una decisione di progetto si conserva e si riusa altrove,
  perché le condizioni cambiano più spesso dei modelli.
- **Ha reso i buchi visibili come buchi.** Una resa non rilevata resta `null`: è più utile sapere che
  `Cohere Rerank 3.5` costa 2 $ per mille interrogazioni e non è stato provato, che ometterlo.

Dalla sezione M2 in avanti il banco viene usato nell'ordine corretto — prima si definiscono i criteri
della decisione, poi si riempie, poi si sceglie — e la prima decisione trattata così è quella del
generatore locale.

## Dove si è arrivati

| | recall@10 | nDCG@10 |
|---|---|---|
| punto di partenza: bge-small da solo | 0.417 | 0.332 |
| + riordino con cross-encoder sui primi 30-50 | 0.478 | 0.387 |
| + candidati costruiti con gte-base | 0.489 | 0.395 |

Il tetto misurato a 200 candidati è 0.82 e il sistema ne converte meno dei due terzi. Le leve provate
e misurate che non hanno reso nulla sono nove: diversificazione MMR, interpolazione dei punteggi,
fusione di riordinatori della stessa famiglia, cascata fra modelli di qualità simile,
multi-formulazione sopra il riordino, riordinatore grande, modello di embedding più forte, quantizzazione
dinamica a 8 bit (più lenta del modello intero), e la lista di candidati più lunga. Le leve che hanno
reso qualcosa sono due: il riordino con cross-encoder, che vale quasi tutto il guadagno del progetto,
e la fusione fra riordinatori di famiglie architetturali diverse, che vale +0.028 di nDCG@10 con
significatività appaiata ma costa venti minuti per domanda su questa macchina.

La conclusione onesta della sezione è che **il divario fra 0.489 e 0.82 non si chiude con gli
strumenti gratuiti su questo hardware**, e che sappiamo quali strumenti lo chiuderebbero e quanto
costano. La verifica di quell'affermazione è progettata in [Appunti11](Appunti11.md), e il costo
stimato è la parte che sorprende.

## Cosa si sottovaluta leggendo solo i numeri finali

Due limiti valgono per ogni cifra di questa sezione. Le annotazioni di FiQA sono **incomplete** —
2,97 fonti annotate per domanda su 61.022 passaggi, mentre il corpus è un forum in cui molte risposte
diverse sono ugualmente valide — quindi ogni punteggio assoluto è sottostimato di una quantità ignota,
e leggendo a mano i risultati contati come errori se ne trovano di pertinenti. E la domanda che il
sistema riceve è **riscritta da un annotatore umano**: in un caso letto a mano in M2 la riscrittura
sposta l'argomento rispetto alle fonti annotate, quindi il recall assorbe anche un disallineamento che
non dipende dal recupero. Entrambi gli effetti spingono nella stessa direzione, verso il basso, e
nessuno dei due è quantificato.
