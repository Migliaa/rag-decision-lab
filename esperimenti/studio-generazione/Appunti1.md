# Appunti 1 (M2) — Dal recupero alla risposta: le decisioni

M2 collega il recupero già costruito in M1 a un modello linguistico che genera la risposta finale.
Le decisioni che contano non riguardano l'implementazione, ma tre scelte di progetto.

**La metrica di M1 non basta a giudicare la generazione.** Recall@10 conta le fonti recuperate,
non le domande servite bene: contando per domanda, un quarto delle 180 non ha mai una fonte
annotata nel contesto, indipendentemente da quanto bene funzioni il generatore. Leggendo a mano
alcuni casi emerge anche un errore di misura più sottile: le annotazioni disponibili sono
incomplete, quindi due domande classificate come "fallimento del recupero" hanno prodotto risposte
corrette lo stesso, perché il contesto conteneva informazione utile anche senza la fonte
specificamente annotata. La posizione della fonte annotata, cioè, non predice se la generazione
riuscirà — una metrica pensata per il recupero mente se applicata alla generazione senza
adattarla.

**Il generatore è stato scelto per costo, non per qualità**: quattro modelli locali erano candidati
inizialmente, ma misurarne la velocità su questa CPU avrebbe richiesto ore per un'informazione che
non si applica ad altri contesti. Un modello via API costa centesimi per l'intera campagna ed è
stato preferito direttamente, documentando il confronto nel banco decisioni invece di eseguirlo.

**Un punteggio medio di recupero non protegge la singola richiesta.** Isolando le domande dove
recupero semplice e selezionato producono contesti diversi, il selezionato — pur migliore in
media — le peggiora in quasi un terzo dei casi. Sostituire una pipeline guardando solo la media
migliora la maggioranza e peggiora silenziosamente una minoranza consistente: prima di farlo in
produzione va misurato quante richieste peggiorerebbero, non solo se la media sale.
