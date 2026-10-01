"""Figure del report finale, ridisegnate per un lettore esterno: niente sigle di metrica, niente
parametri di proiezione nei titoli, etichette in italiano piano. I numeri vengono dagli output
degli esperimenti, non sono ricopiati a mano.

Eseguirlo con: .venv/Scripts/python.exe report/sorgenti/costruisci_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

REPO = Path(__file__).resolve().parents[2]
M1 = REPO / "esperimenti" / "studio-embedding-retrieval" / "output"
M2 = REPO / "esperimenti" / "studio-generazione" / "output"
OUT = REPO / "report" / "figure"
CACHE_HF = REPO / "data" / "cache" / "huggingface"

INCHIOSTRO = "#1f2a30"
GRIGIO = "#c9cfd2"
VERDE = "#2f7d5b"
OCRA = "#d49a3a"
ROSSO = "#b5524a"
BLU = "#3b6e8f"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": INCHIOSTRO,
    "axes.labelcolor": INCHIOSTRO,
    "xtick.color": INCHIOSTRO,
    "ytick.color": INCHIOSTRO,
    "text.color": INCHIOSTRO,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_domanda_fra_i_brani() -> None:
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import TSNE

    ids = json.loads((M1 / "corpus_ids.json").read_text(encoding="utf-8"))
    ex = json.loads((M1 / "query_ranking_example.json").read_text(encoding="utf-8"))
    emb = np.load(M1 / "corpus_embeddings_bge.npy")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                                device="cpu", cache_folder=str(CACHE_HF))
    q = model.encode(["Represent this sentence for searching relevant passages: " + ex["query_text"]],
                     normalize_embeddings=True)[0]
    xy = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(np.vstack([emb, q[None, :]]))
    pts, qxy = xy[:-1], xy[-1]

    gold = set(ex["gold_ids"])
    found = {r["id"] for r in ex["models"]["bge"]["ranking"]}
    is_gold = np.array([c in gold for c in ids])
    is_found = np.array([c in found for c in ids])

    fig, ax = plt.subplots(figsize=(9, 6))
    fondo = ~is_found & ~is_gold
    ax.scatter(pts[fondo, 0], pts[fondo, 1], s=9, color=GRIGIO, linewidths=0, label="altri brani del corpus")
    solo = is_found & ~is_gold
    ax.scatter(pts[solo, 0], pts[solo, 1], s=60, facecolors="none", edgecolors=OCRA, linewidths=1.5,
               label="recuperati, ma non indicati come risposta")
    both = is_found & is_gold
    ax.scatter(pts[both, 0], pts[both, 1], s=80, color=VERDE, edgecolors="white", linewidths=0.8,
               label="recuperati e indicati come risposta")
    ax.scatter([qxy[0]], [qxy[1]], marker="*", s=320, color=INCHIOSTRO, zorder=5,
               label=f"la domanda: “{ex['query_text']}”")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig-1-domanda-fra-i-brani.png", dpi=170)
    plt.close(fig)


def curva_profondita():
    """Riordino della configurazione finale a profondita' diverse, dai punteggi gia' salvati da
    16_configurazione_finale.py: nessun modello da rieseguire."""
    import math
    d = np.load(M1 / "configurazione_finale_candidati.npz", allow_pickle=True)
    gold = [list(x) for x in d["gold_ranks"]]
    sc = np.load(M1 / "configurazione_finale_scores.npy")
    prof = [10, 15, 20, 25, 30, 40, 50, 60]
    tetto, finale = [], []
    for dep in prof:
        tetto.append(np.mean([sum(1 for r in g if r <= dep) / len(g) for g in gold]))
        vals = []
        for i, g in enumerate(gold):
            ordine = np.argsort(-sc[i][:dep], kind="stable")
            pos = {int(q): r + 1 for r, q in enumerate(ordine)}
            fr = [pos[r - 1] if r <= dep else r for r in g]
            vals.append(sum(1 for r in fr if r <= 10) / len(fr))
        finale.append(np.mean(vals))
    return prof, tetto, finale


def fig_scelte_misurate() -> None:
    gte = json.loads((M1 / "gte_scegliere_o_riordinare.json").read_text(encoding="utf-8"))

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})

    # a) la stessa coppia di modelli, secondo la classifica pubblica e misurata qui.
    # I due valori dichiarati vengono dalle schede pubbliche su FiQA (domande singole), vedi Appunti10 di M1.
    dichiarati = [0.403, 0.487]
    misurati = [gte["risultati"]["bge-small da solo"]["ndcg@10"], gte["risultati"]["gte-base da solo"]["ndcg@10"]]
    x = np.array([0, 1])
    w = 0.36
    bd = a.bar(x - w / 2, dichiarati, w, color="white", edgecolor=OCRA, linewidth=1.5, hatch="///",
               label="secondo la classifica pubblica")
    bm = a.bar(x + w / 2, misurati, w, color=BLU, label="misurato su queste domande")
    for r, v in list(zip(bd, dichiarati)) + list(zip(bm, misurati)):
        a.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}".replace(".", ","), ha="center", fontsize=10.5)
    a.set_xticks(x)
    a.set_xticklabels(["modello piccolo", "modello più grande"])
    a.set_ylim(0, 0.6)
    a.set_ylabel("qualità della lista, da 0 a 1")
    a.set_title("La classifica pubblica non si trasferisce", fontsize=12, loc="left")
    a.legend(frameon=False, fontsize=9.5, loc="upper left")

    # b) quanti candidati passare al riordinatore, configurazione finale
    prof, tetto, finale = curva_profondita()
    print("profondita'", prof, "finale", [round(v, 3) for v in finale])
    b.plot(prof, tetto, marker="o", color=GRIGIO, linewidth=2, markeredgecolor=INCHIOSTRO,
           label="fonti giuste presenti fra i candidati")
    b.plot(prof, finale, marker="o", color=VERDE, linewidth=2.5,
           label="fonti giuste nei primi dieci, dopo il riordino")
    b.axvspan(30, 50, color=VERDE, alpha=0.08, linewidth=0)
    b.text(40, 0.42, "da 30 a 50 la differenza\nsta dentro l'incertezza", ha="center", fontsize=9.5)
    b.set_xticks(prof)
    b.set_xlabel("candidati passati al riordinatore")
    b.set_ylabel("quota delle fonti giuste")
    b.set_ylim(0.38, 0.8)
    b.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    b.set_title("Più candidati non vuol dire risultato migliore", fontsize=12, loc="left")
    b.legend(frameon=False, fontsize=9.5, loc="upper left")

    virgola = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    a.yaxis.set_major_formatter(virgola)
    b.yaxis.set_major_formatter(virgola)
    fig.tight_layout(w_pad=3)
    fig.savefig(OUT / "fig-2-scelte-misurate.png", dpi=170)
    plt.close(fig)


def fig_dove_finisce_la_fonte() -> None:
    c = json.loads((M2 / "regimi_recupero.json").read_text(encoding="utf-8"))["conteggi"]
    print("conteggi regimi:", c)
    valori = list(c.values())
    tot = sum(valori)
    etichette = ["in cima alla lista", "fra le prime dieci posizioni", "trovata ma lasciata\nfuori dal contesto", "mai trovata"]
    colori = [VERDE, "#7fb398", OCRA, ROSSO]

    fig, ax = plt.subplots(figsize=(11, 2.3))
    x = 0
    for v, e, col in zip(valori, etichette, colori):
        ax.barh(0, v, left=x, color=col, edgecolor="white", height=0.6)
        ax.text(x + v / 2, 0, f"{round(100 * v / tot)}%", ha="center", va="center", color="white",
                fontsize=12, fontweight="bold")
        ax.text(x + v / 2, -0.52, e, ha="center", va="top", fontsize=9.5)
        x += v
    ax.set_xlim(0, tot)
    ax.set_ylim(-0.95, 0.35)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig-3-dove-finisce-la-fonte.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_scelte_misurate()
    fig_dove_finisce_la_fonte()
    if not (OUT / "fig-1-domanda-fra-i-brani.png").exists():
        fig_domanda_fra_i_brani()
    print("figure scritte in", OUT)
