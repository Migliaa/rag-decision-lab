"""Step 4: disegnare dove cade la query rispetto al suo oro.

Riusa i risultati di 03_query.py (query_ranking_example.json: chi è stato
recuperato, chi è oro) e rifà la proiezione 2D di 02b_visualizza_embedding_tsne.py
includendo stavolta anche la query.

Nota tecnica, importante quanto il disegno: t-SNE non ha un modo di
proiettare un punto nuovo in una mappa già calcolata (non è come PCA, che ha
.transform()). Per mettere la query nella stessa mappa la rifacciamo da capo
con 513 punti (512 passaggi + 1 query) invece di 512, stesso seed e stessa
perplexity di prima. "Stesso seed" vuol dire stesso metodo e riproducibile,
NON vuol dire che ogni passaggio resta esattamente nello stesso pixel di
02b: con un punto in più l'ottimizzazione può spostare leggermente anche gli
altri. Per un solo punto aggiunto su 512 lo spostamento atteso è piccolo, ma
è bene saperlo prima di leggere il disegno come "la stessa mappa di prima".

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/04_figura_query_vicini.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

TSNE_SEED = 0
TSNE_PERPLEXITY = 30  # identica a 02b_visualizza_embedding_tsne.py, per lo stesso motivo

MODELS = {
    "minilm": dict(
        name="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        query_prefix="",
        embeddings_file="corpus_embeddings_minilm.npy",
        title="MiniLM",
    ),
    "bge": dict(
        name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        query_prefix="Represent this sentence for searching relevant passages: ",
        embeddings_file="corpus_embeddings_bge.npy",
        title="BGE-small",
    ),
}


def project_with_query(corpus_embeddings: np.ndarray, query_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rifà il t-SNE su corpus+query insieme. Restituisce (coords_corpus, coord_query)."""
    stacked = np.vstack([corpus_embeddings, query_vector[None, :]])
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=TSNE_SEED, init="pca")
    coords = tsne.fit_transform(stacked)
    return coords[:-1], coords[-1]


def plot_model(ax, coords_2d, query_xy, ids, gold_ids, retrieved_ids, title) -> None:
    gold_ids = set(gold_ids)
    retrieved_ids = set(retrieved_ids)
    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    is_gold = np.array([cid in gold_ids for cid in ids])
    is_retrieved = np.array([cid in retrieved_ids for cid in ids])
    hit = is_gold & is_retrieved  # oro e trovato
    missed = is_gold & ~is_retrieved  # oro ma non trovato in top-k
    false_positive = ~is_gold & is_retrieved  # trovato ma non oro
    background = ~is_gold & ~is_retrieved

    ax.scatter(
        coords_2d[background, 0], coords_2d[background, 1],
        color="lightgray", s=10, alpha=0.4, linewidths=0, label="altro passaggio (non in top-10, non oro)",
    )
    ax.scatter(
        coords_2d[false_positive, 0], coords_2d[false_positive, 1],
        facecolors="none", edgecolors="darkorange", s=70, linewidths=1.3,
        label="in top-10 del modello, MA le qrels non lo segnano rilevante",
    )
    if missed.any():
        ax.scatter(
            coords_2d[missed, 0], coords_2d[missed, 1],
            color="crimson", marker="x", s=90, linewidths=2,
            label="le qrels lo segnano rilevante, MA il modello non l'ha messo in top-10",
        )
    ax.scatter(
        coords_2d[hit, 0], coords_2d[hit, 1],
        color="seagreen", edgecolors="black", s=90, linewidths=0.6,
        label="in top-10 del modello E le qrels lo segnano rilevante",
    )

    # Linee dalla query a OGNI passaggio oro (trovato o no): è la vicinanza
    # che deve essere leggibile, non solo il colore dei punti.
    for cid in gold_ids:
        i = id_to_idx[cid]
        ax.plot(
            [query_xy[0], coords_2d[i, 0]], [query_xy[1], coords_2d[i, 1]],
            color="seagreen" if hit[i] else "crimson", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1,
        )

    ax.scatter([query_xy[0]], [query_xy[1]], color="black", marker="*", s=260, label="query", zorder=5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    ids = json.loads((OUT_DIR / "corpus_ids.json").read_text(encoding="utf-8"))
    results = json.loads((OUT_DIR / "query_ranking_example.json").read_text(encoding="utf-8"))
    query_text = results["query_text"]
    gold_ids = results["gold_ids"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, model_key in zip(axes, MODELS):
        cfg = MODELS[model_key]
        print(f"\n--- {cfg['title']} ---")
        corpus_embeddings = np.load(OUT_DIR / cfg["embeddings_file"])

        model = SentenceTransformer(cfg["name"], revision=cfg["revision"], device="cpu", cache_folder=str(CACHE_HF))
        query_vector = model.encode([cfg["query_prefix"] + query_text], normalize_embeddings=True)[0]

        print("Rifaccio il t-SNE su 513 punti (512 passaggi + la query)...")
        coords_2d, query_xy = project_with_query(corpus_embeddings, query_vector)

        retrieved_ids = [row["id"] for row in results["models"][model_key]["ranking"]]
        plot_model(ax, coords_2d, query_xy, ids, gold_ids, retrieved_ids, cfg["title"])

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=1, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle(
        f"Domanda: \"{query_text}\"  —  t-SNE ricalcolato con la query inclusa (513 punti, seed 0)\n"
        "\"oro\" = qrels del pilota (verità nota); \"top-10\" = i 10 punteggi più alti dati dal modello\n"
        "linea tratteggiata = distanza dalla query a ciascun passaggio oro"
    )
    fig.tight_layout(rect=[0, 0.16, 1, 0.90])

    out_path = OUT_DIR / "query_vicini.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out_path}")


if __name__ == "__main__":
    main()
