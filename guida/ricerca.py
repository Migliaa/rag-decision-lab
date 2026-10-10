"""La ricerca completa del progetto, in una classe: fusione di tre metodi + riordino.

È la configurazione finale degli esperimenti (esperimenti/studio-embedding-retrieval/
16_configurazione_finale.py), scritta in forma semplice. Si usa così:

    ricerca = Ricerca()                   # prepara tutto, una volta (circa un minuto)
    risultati = ricerca.cerca(["domanda"])  # poi ogni domanda costa pochi secondi

Il percorso di una domanda:

    domanda ──► BM25 ──────────► classifica 1 ┐
            ├─► MiniLM (vettori) ► classifica 2 ├─► fusione RRF ─► primi 30 ─► riordino ─► primi 10
            └─► BGE (vettori) ───► classifica 3 ┘                              (cross-encoder)

Perché in due fasi: i tre metodi della prima fase sono veloci e scorrono tutti i
61.022 brani, ma sono approssimativi; il riordinatore è lento e preciso perché
legge insieme domanda e brano, quindi lo si usa solo sui 30 candidati migliori.
"""

import numpy as np

from comuni import CACHE_MODELLI, MODELLI, VETTORI_DIR, carica_corpus, parole, posizioni
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# Il riordinatore: un modello piccolo che dà un voto a ogni coppia (domanda, brano).
RIORDINATORE = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Quanti candidati passano dalla fusione al riordino. Gli esperimenti hanno
# provato 20, 30, 50 e 60: con 30-50 il risultato è migliore e le differenze fra
# queste profondità sono dentro l'incertezza della misura. Qui 30, perché è più
# veloce; oltre 50 il risultato peggiora (candidati in più, ma più rumore).
CANDIDATI = 30

# Costante della fusione RRF (valore standard in letteratura): smorza il peso
# delle primissime posizioni, così un metodo da solo non decide tutto.
COSTANTE_RRF = 60


class Ricerca:
    def __init__(self):
        print("Carico il corpus e costruisco l'indice BM25...")
        self.ids, self.testi = carica_corpus()
        self.bm25 = BM25Okapi([parole(t) for t in self.testi])

        print("Carico i vettori dei brani e i modelli...")
        self.vettori = {}
        self.codificatori = {}
        for nome, impostazioni in MODELLI.items():
            self.vettori[nome] = np.load(VETTORI_DIR / impostazioni["file_vettori"])
            self.codificatori[nome] = SentenceTransformer(
                impostazioni["nome"], revision=impostazioni["revisione"], device="cpu",
                cache_folder=str(CACHE_MODELLI))
        self.riordinatore = CrossEncoder(RIORDINATORE, device="cpu",
                                         cache_folder=str(CACHE_MODELLI), max_length=512)

    def punteggi_vettoriali(self, nome_modello, testo_domanda):
        """Un punteggio per ognuno dei 61.022 brani, con il modello indicato."""
        impostazioni = MODELLI[nome_modello]
        vettore_domanda = self.codificatori[nome_modello].encode(
            [impostazioni["prefisso_domanda"] + testo_domanda], normalize_embeddings=True)[0]
        return self.vettori[nome_modello] @ vettore_domanda

    def solo_bge(self, testo_domanda, quanti=10):
        """Il punto di partenza: BGE da solo, nessuna fusione, nessun riordino."""
        punteggi = self.punteggi_vettoriali("bge", testo_domanda)
        return list(np.argsort(-punteggi, kind="stable")[:quanti])

    def cerca(self, formulazioni, quanti=10):
        """Restituisce le posizioni (nella lista dei brani) dei `quanti` brani migliori.

        `formulazioni` è una lista di testi che dicono la stessa domanda in modi
        diversi (per esempio la domanda riscritta e l'ultima frase dell'utente).
        Usarne più di una trova brani che una sola formulazione perderebbe.
        """
        # --- Fase 1: fusione RRF ------------------------------------------
        # Ogni metodo (BM25, MiniLM, BGE) per ogni formulazione produce una
        # classifica. Ogni brano riceve 1 / (60 + posizione) da ciascuna: un
        # brano che è in alto in molte classifiche somma molto, uno che è in
        # alto in una sola somma poco. Tutti i contributi pesano uguale.
        voci = len(formulazioni) * 3
        fusione = np.zeros(len(self.ids))
        for testo in formulazioni:
            punteggi_bm25 = np.asarray(self.bm25.get_scores(parole(testo)))
            fusione += (1 / voci) / (COSTANTE_RRF + posizioni(punteggi_bm25))
            for nome in MODELLI:
                punteggi = self.punteggi_vettoriali(nome, testo)
                fusione += (1 / voci) / (COSTANTE_RRF + posizioni(punteggi))
        candidati = list(np.argsort(-fusione, kind="stable")[:CANDIDATI])

        # --- Fase 2: riordino ---------------------------------------------
        # Il cross-encoder legge ogni coppia (domanda, brano) e dà un voto.
        # Si usa la prima formulazione, di solito la domanda riscritta.
        coppie = [(formulazioni[0], self.testi[i]) for i in candidati]
        voti = self.riordinatore.predict(coppie, batch_size=32, show_progress_bar=False)
        ordine_nuovo = np.argsort(-voti, kind="stable")[:quanti]
        return [candidati[i] for i in ordine_nuovo]
