"""Step 2b: stessa domanda di 02_visualizza_embedding.py, metodo diverso.

Cambiano due cose rispetto a 02:

1. Riduzione dimensionale: t-SNE invece di PCA. t-SNE cerca di tenere vicini
   nel disegno i punti che erano vicini nello spazio a 384 dimensioni, a
   costo di NON conservare le distanze globali (due gruppi lontani nel
   disegno non sono necessariamente lontani nello spazio vero, e le
   dimensioni dei gruppi nel disegno non sono comparabili). Il vantaggio:
   separa visivamente i gruppi meglio di una proiezione lineare come PCA.
2. Colore: non più cluster KMeans (un raggruppamento automatico, arbitrario
   quanto a numero di gruppi). Qui coloriamo per ARGOMENTO REALE, preso
   dalle qrels del pilota: un passaggio è "oro" per una conversazione se le
   qrels lo segnano come rilevante per quella conversazione. Sono solo 31
   passaggi su 512 (le altre righe del pilota sono distrattori aggiunti per
   riempire il sottoinsieme, senza argomento noto) — il grigio per il resto
   non è un difetto del grafico, è onestà: non inventiamo un argomento per
   passaggi di cui non lo sappiamo.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/02b_visualizza_embedding_tsne.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "output"
QRELS_PATH = REPO_ROOT / "data" / "processed" / "E001-pilot-512" / "qrels.tsv"
REWRITE_PATH = (
    REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa" / "fiqa_rewrite.jsonl"
)

TSNE_SEED = 0
TSNE_PERPLEXITY = 30  # di default in scikit-learn; ragionevole per n=512 (va tenuta sotto n/3)
LABEL_MAX_CHARS = 42


def conversation_id(query_id: str) -> str:
    """L'id di conversazione è la parte prima di '<::>numero_turno'.

    Più turni della stessa conversazione parlano dello stesso argomento:
    raggruppiamo per conversazione, non per singolo turno.
    """
    return query_id.split("<::>")[0]


def load_gold_conversations(qrels_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """corpus_id -> conversation_id, e conversation_id -> turno rappresentativo (il più alto).

    Il secondo dizionario serve solo per scegliere QUALE turno etichettare in legenda:
    deve essere uno dei turni presenti nelle qrels, non un turno qualunque della stessa
    conversazione (che potrebbe non avere nessuna evidenza gold associata).
    """
    corpus_to_conv: dict[str, str] = {}
    conv_to_query_id: dict[str, str] = {}
    with qrels_path.open(encoding="utf-8") as f:
        next(f)  # intestazione: query-id, corpus-id, score
        for line in f:
            query_id, corpus_id, _score = line.rstrip("\n").split("\t")
            conv = conversation_id(query_id)
            if corpus_id in corpus_to_conv and corpus_to_conv[corpus_id] != conv:
                print(f"Attenzione: {corpus_id} e' oro per piu' di una conversazione, tengo la prima incontrata.")
                continue
            corpus_to_conv[corpus_id] = conv
            # Tra i turni con qrels della stessa conversazione, tieni quello più avanzato:
            # di solito è il più esplicito (meno ambiguità di contesto da risolvere).
            turn = int(query_id.split("<::>")[1])
            prev = conv_to_query_id.get(conv)
            if prev is None or turn > int(prev.split("<::>")[1]):
                conv_to_query_id[conv] = query_id
    return corpus_to_conv, conv_to_query_id


def load_conversation_labels(conv_to_query_id: dict[str, str], rewrite_path: Path) -> dict[str, str]:
    """Un'etichetta leggibile per conversazione, presa dalla riscrittura auto-contenuta
    del turno rappresentativo scelto sopra (fiqa_rewrite scioglie i riferimenti che
    fiqa_lastturn lascia impliciti, es. "it" -> "the best formula to evaluate a company")."""
    wanted_query_ids = set(conv_to_query_id.values())
    query_id_to_conv = {v: k for k, v in conv_to_query_id.items()}
    labels: dict[str, str] = {}
    with rewrite_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in wanted_query_ids:
                text = row["text"].replace("|user|:", "").strip()
                if len(text) > LABEL_MAX_CHARS:
                    text = text[: LABEL_MAX_CHARS - 1] + "..."
                labels[query_id_to_conv[row["_id"]]] = text
    return labels


def project_tsne(embeddings: np.ndarray) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=TSNE_SEED,
        init="pca",  # inizializzazione deterministica data la stessa PCA di prima
    )
    return tsne.fit_transform(embeddings)


def plot_model(ax, coords_2d, ids, corpus_to_conv, conv_to_color, conv_to_label, title) -> None:
    is_annotated = np.array([cid in corpus_to_conv for cid in ids])

    # Sfondo: passaggi senza argomento noto in questo pilota, grigio neutro.
    ax.scatter(
        coords_2d[~is_annotated, 0],
        coords_2d[~is_annotated, 1],
        color="lightgray",
        s=12,
        alpha=0.5,
        linewidths=0,
        label="non annotato in questo pilota",
    )

    # Primo piano: i 31 passaggi con argomento noto dalle qrels, un colore per conversazione.
    for conv, color in conv_to_color.items():
        mask = np.array([is_annotated[i] and corpus_to_conv[ids[i]] == conv for i in range(len(ids))])
        if not mask.any():
            continue
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            color=color,
            s=55,
            edgecolors="black",
            linewidths=0.5,
            label=conv_to_label.get(conv, conv),
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    ids = json.loads((OUT_DIR / "corpus_ids.json").read_text(encoding="utf-8"))
    minilm = np.load(OUT_DIR / "corpus_embeddings_minilm.npy")
    bge = np.load(OUT_DIR / "corpus_embeddings_bge.npy")

    corpus_to_conv, conv_to_query_id = load_gold_conversations(QRELS_PATH)
    conv_ids = sorted(set(corpus_to_conv.values()))
    print(f"Passaggi con argomento noto (oro in almeno una conversazione): {len(corpus_to_conv)} su {len(ids)}")
    print(f"Conversazioni distinte rappresentate: {len(conv_ids)}")

    conv_to_label = load_conversation_labels(conv_to_query_id, REWRITE_PATH)
    # tab10 senza il grigio (indice 7): il grigio resta riservato solo ai
    # passaggi "non annotato", non deve ricomparire come colore di categoria.
    tab10_no_gray = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 7]
    if len(conv_ids) > len(tab10_no_gray):
        raise ValueError(f"Servono {len(conv_ids)} colori distinti, tab10 senza grigio ne offre {len(tab10_no_gray)}")
    conv_to_color = {conv: tab10_no_gray[i] for i, conv in enumerate(conv_ids)}

    print("\nEtichette usate in legenda:")
    for conv in conv_ids:
        print(f"  {conv[:12]}...  ->  {conv_to_label.get(conv, '(nessuna riscrittura trovata)')}")

    print("\nProietto MiniLM con t-SNE (puo' richiedere qualche secondo)...")
    minilm_2d = project_tsne(minilm)
    print("Proietto BGE con t-SNE...")
    bge_2d = project_tsne(bge)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6.5))
    plot_model(ax_left, minilm_2d, ids, corpus_to_conv, conv_to_color, conv_to_label, "MiniLM")
    plot_model(ax_right, bge_2d, ids, corpus_to_conv, conv_to_color, conv_to_label, "BGE-small")

    handles, labels_ = ax_left.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.22))
    fig.suptitle(
        "512 passaggi FiQA, proiezione t-SNE 2D (seed 0, perplexity 30)\n"
        "colore = conversazione nota dalle qrels del pilota, grigio = non annotato — schema esplorativo, non ground truth di dominio"
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])

    out_path = OUT_DIR / "embedding_2d_tsne_categorie.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out_path}")

    labels_path = OUT_DIR / "query_topic_labels.json"
    labels_path.write_text(json.dumps(conv_to_label, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Etichette salvate in: {labels_path}")


if __name__ == "__main__":
    main()
