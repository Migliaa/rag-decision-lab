# E001 / D1 — quale rappresentazione recupera le fonti pertinenti?

Stato: in corso; pilot CPU completato, confronto sul corpus completo non ancora eseguito.

## Prima della prova

- **Domanda:** su un campione FiQA congelato, come cambiano qualità e costo fra BM25, MiniLM e BGE-small usando gli stessi passaggi e le stesse query?
- **Perché ci serve:** scegliere la baseline di retrieval e capire quali errori richiedono ricerca lessicale, semantica o gestione della storia.
- **Previsione di Andrea:** BM25 più affidabile per termini e identificatori precisi; BGE favorito sulle richieste semantiche. Il vantaggio BGE/MiniLM sulle altre categorie è da misurare.
- **Precisazione emersa prima della prova:** su follow-up non autonomi il testo dato al retriever può contare più dell'encoder. Il confronto fra ultimo turno e storia sarà separato dal confronto fra modelli.
- **Previsione dell'assistente:** MiniLM e BGE dovrebbero superare BM25 su alcune parafrasi; BM25 può restare competitivo su termini rari, numeri ed eccezioni. Nessun vincitore presunto fra i due encoder.
- **Confronto:** stessa query, stesso corpus, stessi ID e stesso top-k. Cambia soltanto il metodo di ranking: BM25, MiniLM oppure BGE-small con istruzione applicata solo alle query come da model card.
- **Dati:** MTRAG Human, dominio FiQA, corpus a livello di passaggio e qrels ufficiali. Revisione, file, hash e ID del campione saranno scritti in `data/manifest.json` prima della run.
- **Campione CPU:** 128 passaggi deterministici, estendibili a 512 se il primo pilot è sostenibile. Il campione includerà le fonti pertinenti delle query selezionate e distrattori scelti senza guardare i ranking.
- **Misure:** Recall@5/@10, nDCG@10, latenza di codifica e ricerca, memoria del processo, token troncati per modello. I gruppi piccoli di query restano analisi di casi, non percentuali generali.
- **Criterio di stop:** interrompere dopo circa 5 minuti di calcolo senza progresso sufficiente a stimare il costo; nessuna API a pagamento.

## Dopo la prova

- **Run:** `runs/E001-20260912T140703Z-pilot` (128 passaggi, prima prova) e `runs/E001-20260912T141150Z-pilot512` (512 passaggi, riferimento corrente). La seconda run collega un manifest dati immutabile tramite hash.
- **Osservazione sul pilot 512:** Recall@10 BM25 0,583; MiniLM 0,785; BGE 0,625. nDCG@10 rispettivamente 0,377; 0,591; 0,474. Sono medie su 12 query e un corpus costruito con 31 passaggi gold più 481 distrattori deterministici, non punteggi FiQA ufficiali.
- **Costo CPU osservato:** codifica dei 512 passaggi in 10,04 s con MiniLM e 29,50 s con BGE; ricerca media per domanda 1,16 ms, 13,35 ms e 23,81 ms per BM25, MiniLM e BGE. Memoria massima osservata ai punti registrati: 787.255.296 byte, con entrambi gli encoder residenti.
- **Troncamento:** 116/512 testi oltre il limite MiniLM di 256 token; nessun testo oltre il limite BGE di 512, massimo osservato 509. Il vantaggio MiniLM nel pilot non può quindi essere spiegato da una maggiore copertura del testo.
- **Stima, non misura completa:** scalando linearmente la sola codifica a 61.022 passaggi osservati, circa 20 minuti MiniLM e 59 minuti BGE. La run completa dovrà misurare e potrà discostarsi dalla proporzione.
- **Interpretazione:** MiniLM ordina meglio questo sottoinsieme, ma i distrattori campionati rendono il compito più facile e possono favorire un encoder. Le query non autonome mostrano un problema separato: “I am not sure those rules apply to IRAs ...” non espone “SEC pattern day trading” e tutti i metodi falliscono nei primi 10.
- **Decisione:** approfondire; nessun encoder adottato dal pilot. Il confronto successivo usa l'intero corpus oppure un campione esplicitamente più ampio se la misura contraddice la stima. Il confronto ultimo turno/storia resta separato.
- **Apprendimento:** D1; competenze C02–C04.
- **Figura eventuale:** una domanda e tre classifiche prima della proiezione 2D.

