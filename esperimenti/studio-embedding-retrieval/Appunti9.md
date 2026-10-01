# Appunti 9 — Il banco decisioni: lo strumento con cui sono state prese le scelte

Figure: **Fig9.1** = `output/banco_intero.png` (pagina completa, nove decisioni); **Fig9.2** =
`output/banco_unita_indicizzazione.png`; **Fig9.3** = `output/banco_reranker.png`; **Fig9.4** =
`output/banco_reranker_vincolato.png` (stessa tabella con i vincoli «dati riservati» e «budget zero»
attivi). Sorgente: `strumenti/banco-decisioni-rag/banco_decisioni_rag.html` (si apre con
`apri_banco.bat`, contesto tecnico in `CONTESTO_BANCO.md`), catture prodotte da
`18_screenshot_banco.py`. Indice: `INDICE_APPUNTI.md`.

## Perché esiste

Le misure di M1 erano sparse in otto note e diciassette script. Un elenco di risultati non è uno
strumento di decisione: per scegliere servono, affiancati, il guadagno di un'opzione, il suo costo e
le condizioni che la escludono. Il banco raccoglie le nove decisioni incontrate e per ciascuna
mostra solo ciò che serve a prenderla.

## Il principio che ne determina la forma

**Ogni decisione ha criteri propri, quindi colonne proprie.** La prima versione usava una griglia
uniforme con nDCG@10 ovunque, ed era inutilizzabile: l'unità di indicizzazione non si sceglie
guardando il nDCG, che dipende da tutta la pipeline a valle e non dall'unità in sé. Si sceglie
guardando quante unità produce l'indice, quanti token contiene ciascuna, quante risposte finiscono
spezzate fra due unità, e se la ri-segmentazione invalida le annotazioni esistenti (Fig9.2). Il
reranker, al contrario, si sceglie su guadagno rispetto a nessun riordino, latenza per
interrogazione, spesa per mille interrogazioni, parametri e licenza (Fig9.3).

Le colonne in comune sono tre soltanto: il nome dell'opzione, una nota tecnica breve, e l'evidenza.

## La colonna che rende il banco onesto

Ogni riga dichiara da dove viene il suo numero: **misurato** su questo corpus e queste 180 domande,
**dichiarato** dal fornitore o da un benchmark pubblico, **non rilevato**. Senza questa distinzione un
valore preso dalla scheda di un prodotto e uno ottenuto con una misura propria sembrano la stessa
cosa, e non lo sono — i valori pubblicati su BEIR/FiQA vengono da domande diverse e sono più generosi
di circa il 18% rispetto a questo compito conversazionale.

I buchi restano visibili come buchi. Le opzioni a pagamento compaiono con il loro listino e con la
resa vuota: è più utile sapere che `Cohere Rerank 3.5` costa 2 $ per mille interrogazioni e non è
stato provato, che ometterlo o attribuirgli un numero non verificato.

## I vincoli come meccanismo di scelta

Nel lavoro la domanda non è mai quale opzione sia migliore in assoluto, ma quale resti praticabile
date le condizioni. Cinque vincoli si attivano in testa alla pagina — dati che non possono uscire,
budget zero, risposta sotto il secondo, solo CPU, alto volume di interrogazioni — e le opzioni
incompatibili si spengono dichiarando quale vincolo le esclude (Fig9.4). Con «dati riservati» e
«budget zero» insieme, delle quattordici righe del modulo reranker ne restano otto, e le migliori per
latenza spariscono tutte.

## Stato dei dati

Nove decisioni, una settantina di opzioni. Circa un terzo misurato qui, un terzo dichiarato da
fornitori o benchmark, un terzo ancora da rilevare. Il file è autonomo — si apre nel browser senza
alcun server — tiene i dati in `localStorage`, permette di modificare ogni riga e di aggiungere
opzioni, ed esporta in JSON: la stessa struttura si riusa su un altro progetto importando un file
diverso, che è il motivo per cui è stato scritto come strumento invece che come tabella.
