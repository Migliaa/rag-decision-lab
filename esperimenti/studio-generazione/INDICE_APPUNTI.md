# Indice degli Appunti — studio della generazione (M2)

Indice separato da quello di `studio-embedding-retrieval/`, come vuole la convenzione del progetto.
Per il contesto di avvio della sezione leggere [CONTESTO_SESSIONE.md](CONTESTO_SESSIONE.md).

Convenzione: ogni `AppuntiN.md` e' una nota autonoma associata a figure `FigN.X`; i file immagine
restano con il nome prodotto dallo script che li genera ed e' elencato qui. M2 e' una sezione
piccola: le decisioni prese finora sono tenute in una sola nota invece di essere spalmate su piu'
file, a differenza di M1 che per la sua scala ne ha molte.

| Appunti | Argomento | Figure | File immagine | Script |
|---|---|---|---|---|
| [Appunti1.md](Appunti1.md) | Percorso completo di M2: lettura manuale del contesto (regimi di recupero, quarto modo di fallire), scelta del generatore via banco, pilot di generazione con correzione della griglia di giudizio, confronto recupero semplice/selezionato isolato dove conta | Fig1.1 | `output/regimi_recupero.png` | `01_lettura_manuale.py`, `02_figura_regimi.py`, `03_pilot_generazione.py`, `04_domande_dove_differisce.py`, `05_confronto_semplice_selezionato.py` |

Materiale prodotto, non narrato nella nota:

- `output/lettura_manuale.md` — le tre domande con contesto completo, con i punti da annotare
  segnati da `>`.
- `output/pilot_generazione.md`, `output/revisione_pilot.html/.json` — le cinque risposte del
  pilot e il giudizio di Andrea.
- `output/domande_dove_differisce.json`, `output/confronto_semplice_selezionato.md/.json` — il
  confronto recupero semplice/selezionato.
