# D1 — da dove arrivano i dati e che cosa ne abbiamo fatto

## 1. Fonte

I dati provengono dalla repository pubblica IBM `mt-rag-benchmark`, revisione congelata:

`2c618bb98db3c8526433e22d8a2f7320f10a7470`

Sono stati scaricati soltanto:

- corpus FiQA a livello di passaggio;
- query FiQA dell'ultimo turno, con storia e riscrittura in file separati;
- qrels, cioè associazioni fra query e passaggi annotati come pertinenti.

I file originali sono in `data/raw/mtrag-source/`. Il corpus compresso è
`data/raw/mtrag-source/corpora/passage_level/fiqa.jsonl.zip`; la copia estratta è
`data/raw/fiqa/fiqa.jsonl`. Gli originali non vengono riscritti durante gli esperimenti.

Il download è stato eseguito con una copia Git parziale della repository e una selezione
dei soli percorsi FiQA. Su Windows il file ZIP è stato estratto con `Expand-Archive`.

## 2. Formati reali

Il corpus è JSONL: ogni riga è un oggetto JSON indipendente.

```json
{"_id":"10171-0-2129","id":"10171-0-2129","url":"","text":"To add to ...","title":""}
```

Le query sono ancora JSONL:

```json
{"_id":"dc1aaac0b33553d8c897d4150955d803<::>7","text":"|user|: I am not sure those rules apply to IRAs ..."}
```

Le qrels sono TSV: colonne separate da tabulazioni.

```text
query-id                                      corpus-id       score
dc1aa...<::>7                                 146632-0-2180  1
```

`score=1` indica che quel passaggio è annotato come pertinente per quella query. Le qrels
sono usate dalla valutazione, non vengono mostrate al retriever mentre cerca.

## 3. Chi ha scritto le query

Le query vengono da MTRAG Human: sono turni di conversazioni prodotti nel benchmark da
annotatori umani. Non sono state scritte da Andrea o dall'assistente. Dal singolo ID non
attribuiamo la frase a una persona specifica.

Per ogni task di retrieval IBM fornisce tre modi alternativi di formulare la query:

1. **`fiqa_lastturn.jsonl` — ultimo messaggio utente.** È chiamato “last question”
   nella documentazione, ma può essere una frase incompleta, un dubbio o una reazione e
   non deve avere grammaticalmente il punto interrogativo. Esempio:

   ```text
   I am not sure those rules apply to IRAs ...
   ```

2. **`fiqa_questions.jsonl` — tutte le domande dell'utente fino a quel turno.** Per il
   turno 1 contiene Q1; per il turno 2 contiene Q1+Q2; per il turno 3 Q1+Q2+Q3. Perciò
   le frasi precedenti ricompaiono intenzionalmente in molte righe. Ogni riga è un task
   diverso che termina in un turno diverso della stessa conversazione.

3. **`fiqa_rewrite.jsonl` — riscrittura standalone di riferimento.** “Autonoma” qui
   significa comprensibile senza la storia, non generata autonomamente dal nostro
   sistema. Il benchmark risolve riferimenti come “those rules”:

   ```text
   Do the SEC rules on pattern day trading apply to IRAs?
   ```

Sono tre condizioni sperimentali alternative. Possiamo dare al retriever soltanto
l'ultimo turno, la storia delle domande oppure la riscrittura di riferimento e confrontare
le classifiche rispetto alle stesse qrels. Non vanno concatenate tutte insieme. La
riscrittura è un controllo superiore: un sistema reale dovrebbe produrla, e quel lavoro
avrebbe costo ed errori propri.

## 4. Che cos'è il pilot

**Pilot** significa prova preliminare piccola. Non è il nome dello script. Qui abbiamo:

- lo script che prepara la prova: `scripts/prepare_fiqa_pilot.py`;
- i dati prodotti: `data/processed/E001-pilot-512/`;
- la run che usa quei dati: `runs/E001-20260912T141150Z-pilot512/`.

Serve soltanto a controllare che caricamento, modelli, tempi e metriche funzionino prima
di elaborare l'intero corpus. Le sue metriche non misurano FiQA completo.

### Come vengono scelte le 12 query

Il numero 12 è stato scelto dall'assistente come dimensione piccola per la prova tecnica;
non è una decisione di Andrea né una proprietà di IBM. La selezione funziona così:

```text
ID query + stringa fissa
        ↓ SHA-256
impronta numerica sempre uguale
        ↓ ordinamento
primi 12 ID
```

**SHA-256** è una funzione di hash: trasforma un testo in un'impronta di 256 bit,
normalmente mostrata come 64 caratteri esadecimali. Non comprende il significato della
query e non valuta la qualità. Ci serve per ottenere un ordine pseudo-casuale ma
riproducibile degli ID.

Il cosiddetto **seed fisso** è la stringa `E001-D1-fiqa-pilot-v1` aggiunta prima dell'ID.
Usando sempre quella stringa otteniamo sempre le stesse 12 query. Cambiandola cambierebbe
la selezione. Qui non inizializza un generatore casuale: funziona come parte stabile
dell'input dell'hash.

Questo impedisce di scegliere dopo il test soltanto le query su cui il nostro modello si
comporta bene. Non rende però il campione rappresentativo di tutte le categorie.

### Da dove arrivano i 31 passaggi pertinenti

Per ognuna delle 12 query leggiamo tutte le righe corrispondenti nelle qrels. Una query può
avere più di un passaggio pertinente. Unendo gli ID senza duplicati, per queste 12 query
otteniamo 31 passaggi.

Li includiamo tutti nel corpus del pilot per assicurarci che la risposta cercabile sia
presente. Il retriever non riceve l'informazione “questi 31 sono corretti”: vede una lista
mescolata di 512 testi e deve classificarli. Gli altri 481 sono distrattori deterministici.

Questa costruzione rende il pilot utile per verificare la pipeline, ma più facile e meno
realistico della ricerca fra tutti i passaggi FiQA. Non va usata per scegliere il modello
finale.

## 5. Perché le qrels indicano passaggi

La tua idea era quasi corretta: le qrels indicano le **fonti pertinenti** alla query. In
questo benchmark, però, l'unità cercata non è necessariamente un file intero: è un
**passaggio**, cioè una porzione di documento preparata per l'indicizzazione.

La qrel non contiene il testo né una risposta in linguaggio naturale. Contiene soltanto:

```text
ID query → ID passaggio → grado di pertinenza
```

Per leggere il testo dobbiamo cercare quell'ID nel corpus. Più righe per la stessa query
significano che più passaggi sono stati annotati come utili. Possono contenere aspetti
complementari della risposta oppure fonti alternative.
