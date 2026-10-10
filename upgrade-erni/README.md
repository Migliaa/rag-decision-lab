# Upgrade RAG per ERNI: implementazione e apprendimento

Stato al 10 ottobre 2026: tappe A–E realizzate come prototipo e provate (servizio con modello simulato, PostgreSQL con pgvector, container, deployment su Azure Container Apps); le prove e i limiti sono in `EVIDENZE.md`. Il racconto finale (tappa F) e l'eliminazione delle risorse cloud restano da fare. Non è una riscrittura degli esperimenti o una nuova scelta scientifica.

## Risultato circoscritto
Rendere interrogabile una configurazione di retrieval già studiata attraverso un'API Python, con documenti e vettori persistenti, container riproducibile e una prova di deployment Azure. Il laboratorio parte da un campione piccolo esplicito: non sostituisce i risultati sui 61.022 brani.

La preparazione ERNI nell'app Studio è il percorso didattico corrente per questa candidatura, richiesto esplicitamente da Andrea. I vecchi percorsi restano consultabili. Non segnare competenze comprese perché il codice dell'assistente funziona.

## Architettura proposta da motivare insieme
Client → FastAPI → adapter del retrieval esistente → PostgreSQL/pgvector → adapter LLM → validazione → risposta. Log con request_id e durate. In locale Compose avvia API e DB; nel cloud Container Apps ospita l'API, con DB esterno e provider del modello scelto separatamente.

PostgreSQL/pgvector unisce la pratica SQL alla persistenza vettoriale. Partire con ricerca esatta su campione; giustificare eventuali indici approssimati con misure. Non aggiungere Redis, Kubernetes, nuovi agenti o un frontend per questa tappa.

## Tappe e criteri di uscita

| Tappa | L'assistente prepara | Andrea prova e spiega | Evidenza |
|---|---|---|---|
| A · API locale | /health, /answer, schemi, adapter simulati e test | Input valido/invalido, percorso richiesta, errore modello | Contratto e test senza rete |
| B · Retrieval e DB | Adapter della configurazione scelta; import idempotente con provenienza | Query SQL, scelta di k, fonti prima/dopo riavvio | Integrazione su PostgreSQL reale |
| C · Docker | Dockerfile, Compose, env example, istruzioni | Immagine/container/volume; riavvio e modifica configurazione | Avvio ripetibile, niente segreti nell'immagine |
| D · Azure | Disegno risorse, stima e configurazione da revisionare | Account/offerta, costo, accessi, regione | Decisione esplicita prima di risorse a consumo |
| E · Deployment | Distribuzione e raccolta log | Richiesta, errore, revisione, rollback e pulizia | Evidenze del servizio distribuito |
| F · Racconto | Aggiornamento scheda e proposta bullet CV | Due minuti con decisione e limite | Solo affermazioni provate |

## Prima sessione: 45–75 minuti
Leggere nell'app cloud e API, disegnare le responsabilità e scegliere una domanda dal campione esistente. L'assistente realizza il contratto A con provider simulato; Andrea invia tre richieste e guida una modifica al limite dell'input. Prima del collegamento al retrieval si spiega quale configurazione riusare e perché. Non scaricare nuovamente tutti i dati né rilanciare il benchmark completo.

Contratto proposto in contratto.json: esempi, non implementazione. Non inserire chiavi o usare API a pagamento nella tappa A. Prima di B verificare l'ambiente locale (Docker, dipendenze e disponibilità del corpus) senza installazioni globali automatiche.

## Controlli obbligatori
Input vuoto rifiutato prima del modello; JSON del modello invalido gestito; timeout complessivo e retry limitati; fonti restituite appartenenti al contesto; assenza di fonti resa esplicita; importazione ripetuta senza duplicati; dati dopo riavvio; errori sanitizzati e request_id; test simulati separati dalla qualità del retrieval. Per il cloud: accessi limitati, segreti fuori Git, richiesta end-to-end e ritorno a revisione precedente provati.

## Costi e perimetro
Zero spese aggiuntive finché Andrea non decide altrimenti. Valutare l'offerta studenti senza assumere idoneità. I budget Azure avvisano ma non arrestano automaticamente le spese. Controllare separatamente API, DB, registry, modello e log. Nessun endpoint a consumo pubblico senza protezioni.

## Ripresa
Leggere questo file e stato.json. Cominciare da A. EVIDENZE.md registra risultati veri, comando/configurazione e limiti; aggiornare stato tecnico e stato di apprendimento separatamente. Il CV corrente descrive il progetto esistente e non questo piano come risultato.
