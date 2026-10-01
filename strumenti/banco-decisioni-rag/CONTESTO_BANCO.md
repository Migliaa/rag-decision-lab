# Banco decisioni RAG — contesto per l'assistente

Da leggere prima di toccare `banco_decisioni_rag.html`. Scritto perché una sessione futura, dopo una
compattazione del contesto, possa riprenderlo senza rileggere la conversazione in cui è nato.

## Cos'è

Uno strumento per scegliere fra le opzioni di progetto di un sistema RAG. Non è un report e non è una
tabella di risultati: è l'interfaccia che Andrea usa per decidere, e nel report del progetto compare
come *metodo*. Singolo file HTML autonomo, nessun server, nessuna dipendenza: si apre con
`apri_banco.bat` o facendo doppio clic sul file.

Pubblicato anche come artefatto: **https://claude.ai/artifact/SacHvKxqZDXw4SEQ8RAJys**

## Regole di progetto, da non violare

1. **Ogni decisione ha colonne proprie.** Una griglia uniforme è stata provata e scartata: l'unità di
   indicizzazione non si sceglie guardando nDCG@10, che dipende da tutta la pipeline a valle. Si
   sceglie su unità prodotte, token per unità, risposte spezzate fra due unità, validità delle
   annotazioni dopo il taglio. Il reranker si sceglie su guadagno, latenza, spesa, parametri,
   licenza. Se si aggiunge una decisione, si definiscono prima i criteri con cui **quella** scelta si
   prende davvero, poi le colonne.
2. **Niente prosa.** È un'interfaccia per un tecnico: etichette brevi, nessuna spiegazione di cosa sia
   un cross-encoder. Le note per riga sono tecniche e stanno sotto le sei parole.
3. **Scala delle barre assoluta**, dichiarata in intestazione, mai normalizzata sul massimo osservato:
   normalizzando, un valore mediocre sembra eccellente. La tacca grigia marca la riga di riferimento
   (`base: 1`), la colonna Δ è lo scarto da quella.
4. **La colonna evidenza è obbligatoria** e distingue tre stati: `m` misurato su questo corpus, `l`
   dichiarato da fornitore o benchmark pubblico, `u` non rilevato. I valori BEIR/FiQA sono più
   generosi di circa il 18% rispetto al compito conversazionale di MTRAG: non vanno confrontati con i
   nostri come se fossero la stessa scala.
5. **Le opzioni a pagamento restano in tabella** con il loro listino, marcate non verificate. Non si
   omettono perché il progetto non le usa: servono a mostrare che il campo è conosciuto. È una
   richiesta esplicita e permanente di Andrea.
6. **I buchi restano visibili.** Una resa non rilevata si scrive `null`, non si stima.
7. **Costi di natura diversa in colonne diverse.** Ore e denaro non stanno nella stessa casella, e il
   costo per *provare* un'opzione una volta non sta nella stessa colonna del costo a regime: sono i
   numeri di due decisioni diverse — cosa misurare dopo, e cosa mandare in produzione.

## Modifiche del 22 settembre 2026, seconda parte — voce «Generatore» (M2)

Prima decisione di M2 aggiunta al banco, e prima di tutto il progetto presa *con* il banco invece
che registrata a posteriori: criteri decisi prima delle colonne (passo 3 di
`esperimenti/studio-generazione/HANDOVER.md`), poi le colonne, poi le righe. `DATA_VERSION` portata
a `2026-09-22b` per non confondersi con la correzione della mattina sullo stesso file.

Colonne: `par`, `quant`, `tps` (parole/s), `ttft`, `ctx` (finestra), `fit` (dieci passaggi da forum
ci stanno), `absten` (si astiene quando il contesto non basta), `usd180` (costo di una campagna
sulle 180 domande), `vinc`, `lic`, `ev`.

**Nessuna riga ha `ev:"m"`.** I quattro candidati locali (Qwen2.5-3B, Llama-3.2-3B, Phi-3.5-mini,
Qwen2.5-7B) hanno `tps`/`ttft`/`absten` a `null` con `ev:"u"`: la misura di velocità (passo 4) e il
pilot giudicato a mano (passo 5) non sono ancora stati fatti. Le sei righe API (Claude Haiku 4.5 e
Opus 5, GPT-5.4-mini e GPT-5.4, Gemini 3.5 Flash-Lite e 3.1 Pro) hanno `usd180` calcolato dal
listino ufficiale verificato il 22/09/2026 sulla stima di 400.000 token in ingresso e 54.000 in
uscita per 180 domande (Passo 3 dell'handover di M2): è un costo dichiarato da un listino pubblico,
non misurato con una campagna reale, quindi `ev:"l"`. Nessuna riga è oggi evidenziabile come
migliore: la tabella serve a scegliere cosa misurare, non a nascondere che non si è ancora misurato
niente.

Tutte le righe sono taggate `latency` (nessun generatore risponde sotto il secondo) e le righe API
anche `paid`,`external`. `fit` è `true` ovunque: tutte le finestre di contesto elencate superano
abbondantemente i circa 10.000 token di dieci passaggi da forum più domanda e istruzioni — un
controllo fatto, non solo dato per scontato, ma che con questi candidati non discrimina nulla.

## Modifiche del 22 settembre 2026

Andrea ha notato che nel modulo embedding la riga più alta, `bge-large` a 0.450, aveva evidenza
«dichiarato»: il banco violava la propria regola 4 mettendo sulla stessa barra un valore BEIR e i
valori misurati. Corretto in tre punti del codice:

- l'evidenziazione dell'opzione migliore (`pick` in `render()`) considera **solo righe misurate**;
- la colonna Δ mostra lo scarto **solo fra misure**, riga e riferimento entrambi `ev:"m"`;
- le barre dei valori non misurati sono **tratteggiate** (`.sbar.decl`), così un valore dichiarato non
  sembra una misura a colpo d'occhio.

Inoltre `load()` ora confronta `DATA_VERSION` con la versione dello stato salvato nel browser: prima
uno stato salvato nascondeva per sempre i dati di fabbrica aggiornati. **Ogni volta che si modificano
i dati in `FACTORY` va aggiornata `DATA_VERSION`**; lo stato vecchio viene messo da parte in una
chiave di riserva di `localStorage`, non cancellato.

## Modifiche del 20 settembre 2026

Progettare il test a pagamento (`Appunti11.md`) ha mostrato tre lacune, corrette:

- modulo embedding: la colonna di testo `setup`, che mescolava «~5 h CPU» e «0,22 $ · minuti», è
  diventata `seth` (ore) e `setusd` (dollari);
- modulo embedding: nuova colonna `vinc`, *versione bloccabile* — un modello locale si fissa a una
  revisione e resta riproducibile, uno servito può cambiare sotto i piedi;
- modulo reranker: nuova colonna `prova`, *$ per provarlo*, cioè il costo di una singola valutazione
  sulle 180 domande. Le righe «non rilevato» ordinate per questo valore sono la lista della spesa.

Restano proposte e non implementate, con le motivazioni in `Appunti11.md`: sdoppiare il vincolo
«dati riservati» fra «nessun dato esce» e «dati fuori con conservazione zero»; un campo
*interrogazioni al mese attese* nell'intestazione con le colonne di spesa ricalcolate su quello;
una colonna con l'unità di fatturazione (a ricerca o a token), da cui dipende se la profondità di
riordino sia gratuita o a consumo.

## Struttura del codice

Tutto in un file. Le parti che contano:

- `CONSTRAINTS` — i cinque vincoli applicativi (`external`, `paid`, `latency`, `nogpu`, `volume`).
  Una riga con un tag corrispondente si spegne quando quel vincolo è attivo, dichiarando il motivo.
- `FACTORY` — i dati di fabbrica: `project` (intestazione editabile) e `decisions`. Ogni decisione ha
  `t` (titolo), `by` (su cosa si decide, una riga), `cols` (le colonne) e `rows`.
- `cols[].t` — tipo di colonna: `name`, `num` (con `d` = decimali), `pct`, `sec`, `money`, `text`,
  `bool` (con `inv: 1` quando "sì" è la risposta peggiore), `ev`, `delta`, `bar` (con `max`).
  Più colonne possono leggere la stessa chiave `k`: è così che nDCG appare come numero, come barra e
  come Δ.
- `cell()` — formattazione per tipo. `render()` — costruisce le tabelle. Editor in `openEdit()`, che
  genera i campi dalle colonne della decisione, quindi aggiungere un tipo di colonna richiede di
  toccare entrambe.

Persistenza in `localStorage` (chiave `rag-bench-v3`), export/import JSON. Nell'artefatto pubblicato
il salvataggio passa da `window.claude.use("downloads")`, in locale da un link diretto: il codice
prova il primo e ricade sul secondo, non togliere il ramo di riserva. Gli export vanno in `dati/`.

## Come aggiornare l'artefatto pubblicato

Il file non sta più nel percorso da cui è stato pubblicato la prima volta. Per aggiornare lo stesso
URL, passare l'URL esplicitamente allo strumento Artifact insieme al percorso attuale del file,
altrimenti si crea un artefatto nuovo e il link vecchio resta fermo alla versione vecchia.

## Da dove vengono i dati attuali

Sezione M1 del progetto RAG (MTRAG/FiQA, 61.022 passaggi, 180 domande di valutazione, CPU senza GPU).
Le misure sono documentate in `esperimenti/studio-embedding-retrieval/Appunti8.md`; il banco come
metodo in `Appunti9.md`; l'indice completo delle note in `INDICE_APPUNTI.md`. Le catture per il report
si rigenerano con `esperimenti/studio-embedding-retrieval/18_screenshot_banco.py`, che punta a questo
file: se lo si sposta di nuovo, aggiornare il percorso `PAGE` in quello script.

Quando M2 (generazione) e M3 (agenti e wiki) produrranno le loro decisioni, si aggiungono a questo
stesso banco come nuove voci di `decisions`, non in un file separato: il progetto è uno solo in tre
sezioni.
