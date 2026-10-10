# Evidenze dell'upgrade ERNI

Per ogni tappa conservare: versione del codice, configurazione e campione, comando o procedura, risultato osservato, limite. Separare test simulati, integrazione reale, valutazione AI e deployment.

## Tappa A — stato dell'implementazione

### 9 ottobre 2026 · contratto locale con modello simulato

- **Codice:** `servizio/app.py` (schemi ed endpoint), `servizio/modello.py` (recupero e modello simulati), `servizio/__main__.py` (avvio), `tests/test_servizio.py`. Non ancora in un commit.
- **Dipendenze:** extra `servizio` in `pyproject.toml`; installati FastAPI 0.143.0, Pydantic 2.13.5, Uvicorn 0.54.0 nell'ambiente del progetto.
- **Test simulati:** `uv run --extra servizio python -m unittest tests.test_servizio -v` → 9 test superati, senza rete. Coprono: `/health`; risposta valida con `answer`, `sources`, `request_id`; cinque input invalidi → 422; input invalido fermato prima del modello; guasti del modello → 503 e 504 senza dettagli interni nel corpo; fonte non recuperata scartata.
- **Richieste HTTP reali in locale** (`python -m servizio --porta 8011`, eseguite dall'assistente): `GET /health` → 200; `POST /answer` con `What is an IRA?` → 200 con due fonti simulate; `POST /answer` con soli spazi → 422 `string_too_short`. Il log riporta `request_id`, esito e durata.
- **Limiti:** brani e risposta sono inventati e fissi, quindi nessuna di queste prove riguarda la qualità del recupero o della generazione. Il timeout è simulato da un'eccezione: nessuna attesa reale è misurata né interrotta. Il limite di 500 caratteri è provvisorio. Nessuna autenticazione. Il 422 usa il formato predefinito di FastAPI e non contiene `request_id`. Il client di test segnala che `httpx` è deprecato in favore di `httpx2`: non influisce sui risultati.

### 9 ottobre 2026 · recupero reale collegato

- **Configurazione:** `BAAI/bge-small-en-v1.5` (revisione `5c38ec7`), da solo, senza riordino; campione `E001-pilot-512` (512 brani), vettori già calcolati in `data/cache/d1-lab-512/bge.npy`; k = 5 provvisorio. Scelta proposta dall'assistente e adottata quando Andrea ha chiesto di procedere; il riordino resta un passo successivo.
- **Codice:** `servizio/recupero.py`; `python -m servizio` usa il recupero reale, `--recupero simulato` i due brani inventati.
- **Test:** `uv run --extra servizio python -m unittest tests.test_servizio` → 11 superati (9 simulati + 2 di integrazione con modello e vettori locali, senza rete; circa 55 s per il caricamento del modello).
- **Richiesta HTTP reale:** servizio pronto circa 9 s dopo l'avvio. `POST /answer` con `How is 401k account taxed?` (domanda del campione) → 200, cinque fonti; due delle tre fonti annotate per quella domanda sono al primo e al secondo posto (`253391-0-168`, `597574-0-1950`), la terza (`383472-0-727`) non è fra le cinque. Durata registrata nel log: 93 ms la prima richiesta, 37 ms la seconda.
- **Limiti:** una sola domanda, quindi è un esempio e non una misura di qualità; le due durate sono misurazioni singole. Il campione contiene per costruzione tutte le fonti annotate delle sue 12 domande, perciò trovarle fra 512 brani è molto più facile che fra 61.022. La risposta resta simulata: riporta l'inizio del primo brano, nessun testo è generato. Brani e vettori stanno in memoria e vengono ricaricati a ogni avvio (è ciò che la tappa B sostituisce con il database).

## Tappa B — stato dell'implementazione

### 9 ottobre 2026 · PostgreSQL/pgvector in locale

- **Codice e configurazione:** `compose.yaml` (immagine `pgvector/pgvector:pg17`, porta 127.0.0.1:5433, volume `dati-db`), `servizio/db.py` (schema, importazione, ricerca), opzione `--recupero pg` in `servizio/__main__.py`, classe `DatabaseTest` nei test. Driver `psycopg` aggiunto all'extra `servizio`.
- **Dati:** tabella `brani` con 512 righe (id, testo, vettore a 384 dimensioni, campione, modello, revisione); 5 MB su disco. Ricerca esatta con distanza coseno (`<=>`), nessun indice approssimato.
- **Test di integrazione su PostgreSQL reale:** `uv run --extra servizio python -m unittest tests.test_servizio` → 14 superati. I tre nuovi verificano: importazione ripetuta senza duplicati (512 righe prima e dopo); provenienza unica e dichiarata su ogni riga; stessa classifica dei primi 5 fra database e calcolo in memoria per tutte le 12 domande del campione.
- **Persistenza:** dopo `docker compose down` e `docker compose up` (container eliminato e ricreato, volume conservato) la tabella contiene ancora 512 righe, senza reimportare.
- **Richiesta HTTP reale** con `--recupero pg`: stesse cinque fonti della versione in memoria per `How is 401k account taxed?`. Durata nel log: 107 ms la prima richiesta, poi 59, 58 e 56 ms (in memoria: 93 e 37 ms).
- **Limiti:** le durate sono poche misurazioni singole su una domanda, non un confronto di latenza; la differenza indicativa viene dall'apertura di una connessione a ogni richiesta, scelta per semplicità. L'uguaglianza delle classifiche vale per la ricerca esatta su 512 righe: con un indice approssimato o sul corpus intero andrebbe riverificata. La password del database è un valore di solo sviluppo scritto in `compose.yaml`; i segreti si trattano nella tappa C. Il servizio gira ancora fuori da Docker. Non in un commit.

## Tappa C — stato dell'implementazione

### 9 ottobre 2026 · servizio in container

- **File:** `Dockerfile`, `.dockerignore` (ammette solo `servizio/` e `src/`), `servizio/requirements-immagine.txt` (versioni dell'ambiente locale, torch per sola CPU), `compose.yaml` (servizi `db`, `api`, `importa`), `.env.example`. `.env` con password generata, escluso da Git (verificato con `git check-ignore`).
- **Immagine `rag-erni-api`:** Python 3.14-slim, dipendenze, codice, modello `bge-small-en-v1.5` alla revisione fissata scaricato in costruzione; 2,21 GB. Processo eseguito dall'utente non amministratore `servizio`.
- **Avvio:** `docker compose up -d --build`, poi `docker compose run --rm importa` la prima volta (512 righe). `api` e `db` risultano `healthy`; api su 127.0.0.1:8020, db su 127.0.0.1:5433.
- **Richieste HTTP al container:** `/health` 200; `POST /answer` con `How is 401k account taxed?` → stesse cinque fonti delle tappe A e B; domanda di soli spazi → 422. Durata nel log: 51, 35, 35 ms.
- **Niente segreti nell'immagine:** la password non compare nella cronologia degli strati (`docker history`), nelle variabili dell'immagine, nei file di `/app`; un container avviato dall'immagine senza Compose non ha `DATABASE_URL`. Nell'immagine non ci sono dati (`/app/data` contiene solo la cache del modello).
- **Configurazione senza ricostruire:** con `RECUPERO_K=3` il servizio restituisce tre fonti, tornando a 5 ne restituisce cinque; l'identificativo dell'immagine è lo stesso prima e dopo.
- **Riavvio:** dopo `docker compose down` e `docker compose up -d` il servizio risponde con cinque fonti senza reimportare.
- **Test dal computer:** 14 superati dopo il passaggio della password a `.env`.
- **Limiti:** immagine pesante per via di torch; nessun tentativo di ridurla. Le durate sono misurazioni singole. La password è cambiata rispetto alla tappa B, quindi il volume è stato ricreato e i brani reimportati. Il database è ancora un container locale: nel cloud sarà un servizio esterno. Nessuna autenticazione sull'API. `DATABASE_URL` è visibile a chi può ispezionare il container in esecuzione. Non in un commit.

## Tappa D — stato dell'implementazione

### 10 ottobre 2026 · prezzi verificati, nessuna risorsa creata

- Andrea dichiara un abbonamento Azure Student da 200 € per 30 giorni (i crediti scadono dopo 30 giorni dall'attivazione); non verificato da qui. Regione scelta da Andrea: West Europe. Azure CLI 2.91.0 installata con winget su richiesta di Andrea; login non eseguito. Nessuna risorsa cloud creata, spesa 0.
- **Prezzi di listino letti dall'API pubblica dei prezzi Azure** (West Europe, EUR, 10 ottobre 2026): Container Registry Basic 0,1466 € al giorno (circa 4,4 € al mese) con 10 GB di archiviazione inclusi, l'immagine del servizio pesa 2,21 GB; Container Apps, piano a consumo: 180.000 vCPU-secondi, 360.000 GiB-secondi e 2 milioni di richieste gratuiti al mese per abbonamento, riduzione a zero repliche possibile, 0,49 € per milione di richieste oltre la quota.
- **Non verificato:** prezzo orario di PostgreSQL Flexible Server Burstable B1ms e dello spazio disco (la query all'API non ha restituito voci), prezzo per secondo oltre la quota gratuita di Container Apps (l'API lo arrotonda a zero), costo di Log Analytics. Da leggere nel calcolatore prezzi del portale con regione e valuta corretti prima di creare il database.
- pgvector su PostgreSQL Flexible Server: l'estensione va prima abilitata nel parametro `azure.extensions`, poi creata con `CREATE EXTENSION` (documentazione Microsoft, aggiornata luglio 2026).

### 10 ottobre 2026 · fase 1: servizio in Container Apps (senza database)

- **Abbonamento:** letto da `az rest`: offerta `FreeTrial` con limite di spesa attivo, quindi non Student come dichiarato da Andrea. I crediti non si possono superare; esauriti o scaduti, Azure disattiva le risorse. Saldo e data di scadenza non letti da qui.
- **Risorse create** (West Europe, gruppo `rg-rag-erni`, etichette `progetto=rag-erni`): registro Basic `ragerni30442`, ambiente `cae-rag-erni` (con workspace Log Analytics creato in automatico), app `ca-rag-api` con 1 vCPU, 2 GiB, da 0 a 1 replica, ingresso pubblico limitato all'IP di Andrea (regola `solo-andrea`), immagine scaricata con identità gestita (nessuna password per il registro).
- **Immagine:** variante `Dockerfile.cloud` (immagine locale più i soli due file di dati, circa 1 MB), pubblicata come `v1` e `v2`; caricamento di `v1` in 154 s. Il servizio usa la ricerca in memoria con BGE.
- **Richieste dal computer di Andrea all'indirizzo pubblico:** `/health` 200; `POST /answer` 200 con le stesse cinque fonti delle prove locali; domanda di soli spazi e corpo vuoto: 422.
- **Avvio a freddo da zero repliche:** 30,4 s nella prima prova (revisione `v1`), 4,7 s con la revisione `sana`; sono misure singole e la differenza non è spiegata.
- **Errore provocato e revisioni:** `GUASTO_MODELLO` (nuova variabile d'ambiente del servizio) fa fallire il modello simulato. Revisione `guasto2`: 503 `modello_non_disponibile` con `request_id`, anche a caldo; il `request_id` `05234755…` compare nei log di Azure (query su Log Analytics) con il dettaglio interno `connessione rifiutata da modello.interno:8080`, che al cliente non è arrivato. Ritorno alla revisione `sana` con un comando di spostamento del traffico: risposta 200 immediata.
- **Incidenti:** (1) passando `--args` con un trattino a `az containerapp update`, gli argomenti sono stati salvati come un'unica stringa e la revisione `v2` non partiva (Python: modulo non trovato); il traffico era stato spostato su di essa automaticamente e l'ho riportato su `v1` con un comando (durata dell'interruzione non misurata). (2) Le revisioni create dopo `v2` hanno ereditato il comando sbagliato e non sono mai diventate sane; due richieste in quel periodo hanno atteso 240 s senza risposta. Risolto svuotando comando e argomenti. (3) Ho impostato per errore `GUASTO_MODELLO=nessuno` nella revisione `pulita`, valore che il servizio rifiuta all'avvio; rimosso nella revisione `sana`.
- **Limiti:** l'ingresso è aperto solo a un indirizzo IP, quindi il servizio non è raggiungibile dal telefono. Nessuna autenticazione nel codice. Costi effettivi non ancora letti in Gestione costi (i dati arrivano con ritardo di ore). Le revisioni `sana` e `guasto2` sono attive, le altre disattivate. Non in un commit.

### 10 ottobre 2026 · fase 2: PostgreSQL con pgvector su Azure

- **Server:** `pgragerni93284`, PostgreSQL 16.15, Flexible Server livello Burstable `Standard_B1ms`, 32 GB di disco, senza alta disponibilità, backup a 7 giorni, **North Europe**. Il servizio resta in West Europe: i due sono in regioni diverse perché West Europe ha rifiutato la creazione (`The location is restricted from performing this operation`), un limite dell'offerta di prova; North Europe è stata la prima delle regioni provate che l'ha accettata. La CLI segnala che B1ms è a pagamento (livello `Paid Tier`); il prezzo orario non è stato letto da qui e non è noto se rientri nei servizi gratuiti dell'offerta.
- **Configurazione:** database `rag`; l'estensione `vector` è nell'elenco `azure.extensions` e creata dall'importazione (pgvector 0.8.2); connessione cifrata verificata (`ssl` attivo). Accesso pubblico abilitato con due regole: l'IP di Andrea e la regola speciale `0.0.0.0` «servizi Azure». Le regole per singolo indirizzo non erano praticabili perché l'app in Container Apps ha 321 indirizzi in uscita.
- **Importazione dal computer di Andrea nel database in cloud:** due esecuzioni consecutive, 512 righe e 512 id distinti entrambe le volte, provenienza unica (`E001-pilot-512`, `bge-small-en-v1.5`).
- **Servizio collegato:** l'URL di connessione (con la password) è un segreto dell'app (`db-url`), esposto come variabile `DATABASE_URL` con `secretref`; la modalità di ricerca è `RECUPERO_MODALITA=pg`; immagine `v3`. Le richieste dalla rete di Andrea alla revisione `pg` restituiscono 200 con le stesse cinque fonti delle prove in memoria e in locale. Durate misurate dal client: 4,1 s al primo accesso (replica avviata a freddo) e 0,4 s a caldo (tre misure); in locale con il database in un container erano circa 0,06 s lato server, ma le due misure non sono confrontabili (client lontano, app e database in regioni diverse, una connessione nuova per richiesta).
- **Persistenza:** riavvio del server database (64 s); poi 512 righe e il servizio risponde 200 con le stesse cinque fonti.
- **Modifiche al codice:** `RECUPERO_MODALITA` e `GUASTO_MODELLO` rendono modalità di ricerca e guasto configurabili da variabile d'ambiente; `Dockerfile.cloud` non passa più `--recupero` al comando di avvio.
- **Incidenti, in ordine:** opzioni della CLI non valide (`--high-availability`, `--database-name` che vale solo per cluster elastici); provider `Microsoft.DBforPostgreSQL` non registrato; regione West Europe bloccata; server creato con `--public-access None` che disattiva l'accesso pubblico (riabilitato con `update --public-access Enabled`); nome del database con `--name` invece di `--database-name`. Nessuna risorsa a pagamento è rimasta da tentativi falliti.
- **Limiti e scelte da dichiarare:** la regola `0.0.0.0` ammette qualsiasi risorsa Azure di qualunque cliente a tentare la connessione (servono comunque password e TLS); in produzione si userebbero rete privata o endpoint privato, che hanno un costo aggiuntivo. L'app usa l'utente amministratore del server; in produzione servirebbe un utente con permessi minimi o l'autenticazione con identità gestita. Latenza fra regioni non misurata separatamente. Costi non ancora letti. Il server è acceso: costa finché non viene fermato o eliminato.

## Tappa A — stato dell'apprendimento

Annotare spiegato / provato insieme / spiegato da Andrea soltanto con evidenza della sessione. Non dedurlo dai test.

| Argomento | Stato | Evidenza |
|---|---|---|
| Scopo dell'upgrade; che cosa sono API, HTTP, FastAPI, Pydantic | spiegato | Il 9 ottobre Andrea dichiara di non conoscerli; spiegazione da parole piane data lo stesso giorno e riassunta in `SPIEGAZIONE.md`; nessuna prova o ripetizione di Andrea |
| Percorso client → richiesta HTTP → API → logica → risposta | non affrontato | Prima spiegazione non compresa: dava per noti i termini della riga sopra |
| Validazione dell'input e 422 | non affrontato | come sopra |
| Guasto del modello: 503/504, errore senza dettagli interni, `request_id` | non affrontato | come sopra |
| Modifica del limite della domanda con verifica | non affrontato | — |
