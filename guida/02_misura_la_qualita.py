"""Esempio 2 — Come si misura se la ricerca funziona: recall, nDCG e intervallo di confidenza.

Cosa fa:
  - PARTE A spiega le due misure su un esempio piccolo a mano;
  - PARTE B le calcola davvero per BGE da solo su 180 domande e 61.022 brani.
    Il risultato deve coincidere con la riga "bge-small da solo" del README
    (recall@10 circa 0.417, nDCG@10 circa 0.332).

Come si lancia (dalla cartella "rag"; la parte B impiega circa un minuto):
    uv run python guida/02_misura_la_qualita.py

Corrisponde a: esperimenti/studio-embedding-retrieval/10_baseline_180.py
"""

import math

import numpy as np

from comuni import (CACHE_MODELLI, MODELLI, VETTORI_DIR, carica_corpus, carica_domande,
                    carica_fonti_giuste, posizioni)
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# LE DUE MISURE
# Per ogni domanda si sa quali brani sono "giusti" (le fonti annotate a mano).
# Dopo la ricerca si guarda in che posizione della classifica sono finiti.
# ---------------------------------------------------------------------------
def recall_a(posizioni_giuste, k):
    """Quota delle fonti giuste che compare fra i primi k risultati."""
    trovate = [p for p in posizioni_giuste if p <= k]
    return len(trovate) / len(posizioni_giuste)


def ndcg_a(posizioni_giuste, k):
    """Come recall, ma premia le fonti giuste che stanno più in alto.

    Una fonte in prima posizione vale 1, in seconda 0,63, in decima 0,29, oltre
    k vale 0. Il totale è diviso per il massimo ottenibile (tutte le fonti
    giuste nelle prime posizioni), quindi il risultato va da 0 a 1.
    """
    ottenuto = sum(1 / math.log2(p + 1) for p in posizioni_giuste if p <= k)
    massimo = sum(1 / math.log2(i + 1) for i in range(1, min(len(posizioni_giuste), k) + 1))
    return ottenuto / massimo


def intervallo_di_confidenza(valori, ripetizioni=2000):
    """Quanto può oscillare la media se le domande fossero state altre.

    Si estraggono 2000 volte 180 domande a caso (con ripetizione) fra quelle
    disponibili, si calcola ogni volta la media, e si tengono i valori fra il
    2,5% e il 97,5% di queste medie. È l'incertezza dovuta al campione.
    """
    valori = np.array(valori)
    generatore = np.random.default_rng(0)
    scelte = generatore.integers(0, len(valori), size=(ripetizioni, len(valori)))
    medie = valori[scelte].mean(axis=1)
    return np.percentile(medie, 2.5), np.percentile(medie, 97.5)


# ---------------------------------------------------------------------------
# PARTE A — Un esempio inventato: due fonti giuste, arrivate in posizione 2 e 15.
# ---------------------------------------------------------------------------
esempio = [2, 15]
print("PARTE A (esempio inventato): fonti giuste arrivate in posizione 2 e 15")
print(f"  recall@10 = {recall_a(esempio, 10):.2f}   (una fonte su due è nei primi 10)")
print(f"  nDCG@10   = {ndcg_a(esempio, 10):.2f}   (la fonte in seconda posizione vale 0,63 su un massimo di 1,63)\n")

# ---------------------------------------------------------------------------
# PARTE B — La misura vera: BGE da solo su 180 domande.
# ---------------------------------------------------------------------------
print("PARTE B: BGE-small da solo, 180 domande, 61.022 brani")
ids, _ = carica_corpus()
posto_del_brano = {id_brano: i for i, id_brano in enumerate(ids)}  # id -> riga nella matrice

fonti_giuste = carica_fonti_giuste()
domande = carica_domande("fiqa_rewrite")  # la domanda riscritta, comprensibile da sola
id_domande = sorted(fonti_giuste)

bge = MODELLI["bge"]
vettori_brani = np.load(VETTORI_DIR / bge["file_vettori"])  # [61.022 x 384], già calcolata
modello = SentenceTransformer(bge["nome"], revision=bge["revisione"], device="cpu",
                              cache_folder=str(CACHE_MODELLI))
vettori_domande = modello.encode([bge["prefisso_domanda"] + domande[q] for q in id_domande],
                                 normalize_embeddings=True, batch_size=32, show_progress_bar=False)

recall_per_domanda = []
ndcg_per_domanda = []
for numero, id_domanda in enumerate(id_domande):
    punteggi = vettori_brani @ vettori_domande[numero]  # un punteggio per ogni brano
    classifica = posizioni(punteggi)
    posizioni_giuste = [int(classifica[posto_del_brano[g]])
                        for g in fonti_giuste[id_domanda] if g in posto_del_brano]
    recall_per_domanda.append(recall_a(posizioni_giuste, 10))
    ndcg_per_domanda.append(ndcg_a(posizioni_giuste, 10))

basso, alto = intervallo_di_confidenza(recall_per_domanda)
print(f"  domande valutate: {len(id_domande)}")
print(f"  recall@10 = {np.mean(recall_per_domanda):.3f}   (intervallo di confidenza 95%: {basso:.3f} - {alto:.3f})")
print(f"  nDCG@10   = {np.mean(ndcg_per_domanda):.3f}")
print("\nL'intervallo è largo: differenze più piccole di circa 0,05 fra due configurazioni")
print("possono essere solo effetto delle domande scelte, per questo gli esperimenti usano tutte le 180.")
