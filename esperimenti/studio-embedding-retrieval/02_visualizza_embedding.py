"""Step 2: guardare lo spazio degli embedding prima di passare alle query.

Non cerca ancora niente (le query sono lo step successivo). Prende le due
matrici già salvate (MiniLM in 01_indicizzazione.py, BGE in
01b_indicizzazione_bge.py) e le rende visibili in 2 dimensioni.

Due avvertenze da tenere presenti leggendo il grafico, non solo da premettere:

1. PCA riduce 384 dimensioni a 2. Si perde informazione per costruzione, e
   quanta se ne perde va MISURATA (la varianza spiegata stampata sotto), non
   assunta. Se le prime 2 componenti spiegano poco, due punti vicini nel
   disegno potrebbero non essere i più vicini nello spazio vero a 384
   dimensioni: il disegno è indicativo, non la geometria reale.
2. I colori vengono da KMeans, un raggruppamento automatico dei vettori
   vicini tra loro: sono cluster trovati nello spazio, non argomenti
   etichettati da una persona (il corpus FiQA che usiamo non ha categorie).
   MiniLM e BGE vivono in spazi diversi: lo stesso colore nei due grafici
   NON indica lo stesso gruppo di passaggi, ogni grafico va letto per conto
   suo.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/02_visualizza_embedding.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # nessuna finestra grafica: salviamo direttamente su file
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

OUT_DIR = Path(__file__).resolve().parent / "output"
N_CLUSTERS = 8  # scelta arbitraria di partenza, non derivata dai dati


def reduce_and_cluster(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Da [n_passaggi, 384] a coordinate 2D + etichetta di cluster.

    Restituisce anche la varianza spiegata dalle 2 componenti scelte: è la
    frazione della "dispersione" originale dei 384 numeri che sopravvive nel
    disegno. Bassa non vuol dire che il metodo è sbagliato: vuol dire che il
    disegno racconta meno della geometria reale.
    """
    pca = PCA(n_components=2, random_state=0)
    coords_2d = pca.fit_transform(embeddings)
    explained = float(pca.explained_variance_ratio_.sum())

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)  # clustering sui 384 numeri originali, non sui 2D

    return coords_2d, cluster_labels, explained


def plot_model(ax, coords_2d: np.ndarray, cluster_labels: np.ndarray, title: str, explained: float) -> None:
    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=18,
        alpha=0.75,
        linewidths=0,
    )
    ax.set_title(f"{title}\nvarianza spiegata dalle 2 componenti: {explained:.1%}")
    ax.set_xlabel("componente principale 1")
    ax.set_ylabel("componente principale 2")
    return scatter


def main() -> None:
    minilm = np.load(OUT_DIR / "corpus_embeddings_minilm.npy")
    bge = np.load(OUT_DIR / "corpus_embeddings_bge.npy")
    print(f"MiniLM: {minilm.shape}, BGE: {bge.shape}")

    minilm_2d, minilm_clusters, minilm_explained = reduce_and_cluster(minilm)
    print(f"MiniLM - varianza spiegata (2 componenti): {minilm_explained:.1%}")

    bge_2d, bge_clusters, bge_explained = reduce_and_cluster(bge)
    print(f"BGE    - varianza spiegata (2 componenti): {bge_explained:.1%}")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.5))
    plot_model(ax_left, minilm_2d, minilm_clusters, "MiniLM", minilm_explained)
    plot_model(ax_right, bge_2d, bge_clusters, "BGE-small", bge_explained)
    fig.suptitle(
        f"512 passaggi FiQA, proiezione PCA 2D, colore = cluster KMeans (k={N_CLUSTERS} per modello, indipendenti tra i due grafici)"
    )
    fig.tight_layout()

    out_path = OUT_DIR / "embedding_2d_pca.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigura salvata in: {out_path}")


if __name__ == "__main__":
    main()
