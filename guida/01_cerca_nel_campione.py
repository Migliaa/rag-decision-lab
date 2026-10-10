"""Esempio 1 — Come si cerca: dal testo al vettore, dal vettore alla classifica.

Cosa fa: prende una domanda e cerca i 5 brani più adatti fra 512, in due modi:
  - BM25: conta le parole in comune (ricerca "per parole");
  - BGE: confronta il significato, trasformando i testi in vettori (ricerca "per significato").

Come si lancia (dalla cartella "rag"):
    uv run python guida/01_cerca_nel_campione.py
    uv run python guida/01_cerca_nel_campione.py "What is an IRA?"

Corrisponde a: src/retrieval.py e laboratorio/d1_query.py (gli originali).
"""

import sys

import numpy as np
from rank_bm25 import BM25Okapi

from comuni import (CAMPIONE_DIR, CACHE_MODELLI, MODELLI, VETTORI_CAMPIONE_DIR,
                    leggi_jsonl, parole, testo_del_brano)
from sentence_transformers import SentenceTransformer

bge = MODELLI["bge"]
modello = SentenceTransformer(bge["nome"], revision=bge["revisione"], device="cpu",
                              cache_folder=str(CACHE_MODELLI))

# ---------------------------------------------------------------------------
# PARTE A — Che cos'è un vettore. Tre frasi inventate per vedere il meccanismo.
# ---------------------------------------------------------------------------
frasi = [
    "I put money into a retirement account",
    "A pension fund holds savings for old age",
    "The football match ended in a draw",
]

# encode() trasforma ogni frase in una lista di 384 numeri: il suo "vettore".
# normalize_embeddings=True porta ogni vettore a lunghezza 1; così il prodotto
# fra due vettori (@) dice quanto puntano nella stessa direzione, cioè quanto
# le frasi si somigliano di significato: 1 = identiche, vicino a 0 = estranee.
vettori = modello.encode(frasi, normalize_embeddings=True)
print("Forma della matrice:", vettori.shape, "-> 3 frasi, 384 numeri ciascuna\n")

somiglianza = vettori @ vettori.T  # ogni frase confrontata con ogni altra
for i in range(len(frasi)):
    for j in range(i + 1, len(frasi)):
        print(f"{somiglianza[i][j]:.2f}  {frasi[i]!r}  <->  {frasi[j]!r}")
print("\nLe prime due parlano della stessa cosa con parole diverse e hanno il punteggio più alto.")
print("Anche frasi estranee restano intorno a 0,5: con BGE conta l'ordine, non il valore assoluto.\n")

# ---------------------------------------------------------------------------
# PARTE B — La ricerca vera, sui 512 brani del campione.
# ---------------------------------------------------------------------------
domanda = sys.argv[1] if len(sys.argv) > 1 else "How is 401k account taxed?"
print(f"DOMANDA: {domanda}\n")

brani = leggi_jsonl(CAMPIONE_DIR / "corpus.jsonl")
ids = [b["_id"] for b in brani]
testi = [testo_del_brano(b) for b in brani]


def mostra(titolo, punteggi):
    """Stampa i 5 brani con il punteggio più alto."""
    migliori = np.argsort(-punteggi, kind="stable")[:5]  # indici, dal migliore
    print(f"=== {titolo} ===")
    for posto, i in enumerate(migliori, start=1):
        anteprima = " ".join(testi[i].split())[:130]
        print(f"{posto}. [{ids[i]}] punteggio {punteggi[i]:.3f}  {anteprima}")
    print()


# 1) BM25: l'indice conta, per ogni parola, in quali brani compare e quanto è rara.
#    Si costruisce una volta; poi ogni domanda è solo un calcolo di punteggi.
bm25 = BM25Okapi([parole(t) for t in testi])
mostra("BM25 (parole in comune)", np.asarray(bm25.get_scores(parole(domanda))))

# 2) BGE: i vettori dei 512 brani sono già stati calcolati e salvati (bge.npy).
#    Per la domanda si calcola solo il suo vettore e si fa un unico prodotto
#    con la matrice [512 brani x 384 numeri]: restituisce 512 punteggi.
vettori_brani = np.load(VETTORI_CAMPIONE_DIR / "bge.npy")
vettore_domanda = modello.encode([bge["prefisso_domanda"] + domanda], normalize_embeddings=True)[0]
mostra("BGE (significato)", vettori_brani @ vettore_domanda)
