"""Step 6: generalizzare da una domanda a dodici, e aggiungere BM25.

03/04/05 mostravano il meccanismo su UNA domanda. Qui si ripete lo stesso
calcolo (recall@10, nDCG@10) sulle 12 domande del pilota, con tre metodi:

- BM25: ricerca lessicale (conteggio di parole in comune, pesato per
  rarità). Nessun embedding, nessuna delle 384 dimensioni viste finora.
- MiniLM e BGE: gli stessi embedding già calcolati in 01/01b.

Punto importante corretto rispetto a 02b: l'oro qui è quello ESATTO del
singolo turno/domanda (dalle qrels), non quello unito per intera
conversazione — due turni della stessa conversazione possono avere oro
diverso (vedi Appunti1, sezione limiti).

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/06_metriche_12_domande.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "E001-pilot-512" / "corpus.jsonl"
QRELS_PATH = REPO_ROOT / "data" / "processed" / "E001-pilot-512" / "qrels.tsv"
REWRITE_PATH = (
    REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa" / "fiqa_rewrite.jsonl"
)
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

K = 10

MODELS = {
    "minilm": dict(
        name="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        query_prefix="",
        embeddings_file="corpus_embeddings_minilm.npy",
    ),
    "bge": dict(
        name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        query_prefix="Represent this sentence for searching relevant passages: ",
        embeddings_file="corpus_embeddings_bge.npy",
    ),
}


def indexed_text(passage: dict) -> str:
    title = passage.get("title", "")
    text = passage["text"]
    return f"{title}\n{text}" if title else text


def lexical_tokens(text: str) -> list[str]:
    """Stessa tokenizzazione di src/retrieval.py: minuscolo, conserva
    identificatori tipo ZX-401 e numeri decimali come un solo token."""
    return re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower())


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["_id"])
            texts.append(indexed_text(row))
    return ids, texts


def load_qrels(path: Path) -> dict[str, set[str]]:
    gold_by_query: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            query_id, corpus_id, _score = line.rstrip("\n").split("\t")
            gold_by_query.setdefault(query_id, set()).add(corpus_id)
    return gold_by_query


def load_rewrite_texts(query_ids: set[str], path: Path) -> dict[str, str]:
    """Testo auto-contenuto della domanda (scioglie riferimenti impliciti
    tipo "it"), stesso motivo già spiegato in 02b/03: non usiamo la frase
    isolata (fiqa_lastturn), ambigua fuori dal suo turno di conversazione."""
    texts = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in query_ids:
                texts[row["_id"]] = row["text"].replace("|user|:", "").strip()
    return texts


def dcg_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank, cid in enumerate(ranked_ids[:k], start=1) if cid in gold_ids)


def idcg_at_k(n_gold: int, k: int) -> float:
    n = min(n_gold, k)
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, n + 1))


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    return len(set(ranked_ids[:k]) & gold_ids) / len(gold_ids)


def ndcg_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    idcg = idcg_at_k(len(gold_ids), k)
    return dcg_at_k(ranked_ids, gold_ids, k) / idcg if idcg else 0.0


def rank_bm25(bm25: BM25Okapi, query_text: str, ids: list[str]) -> list[str]:
    scores = np.asarray(bm25.get_scores(lexical_tokens(query_text)))
    order = np.argsort(-scores, kind="stable")
    return [ids[i] for i in order]


def rank_dense(corpus_embeddings: np.ndarray, query_vector: np.ndarray, ids: list[str]) -> list[str]:
    scores = corpus_embeddings @ query_vector
    order = np.argsort(-scores, kind="stable")
    return [ids[i] for i in order]


def main() -> None:
    ids, texts = load_corpus(CORPUS_PATH)
    saved_ids = json.loads((OUT_DIR / "corpus_ids.json").read_text(encoding="utf-8"))
    assert ids == saved_ids, "L'ordine del corpus letto qui non coincide con quello usato per gli embedding salvati"

    gold_by_query = load_qrels(QRELS_PATH)
    query_ids = sorted(gold_by_query)  # le 12 domande del pilota, una per turno
    print(f"Domande nel pilota: {len(query_ids)}")

    query_texts = load_rewrite_texts(set(query_ids), REWRITE_PATH)

    print("Costruisco l'indice BM25 (una tantum, come l'embedding del corpus)...")
    bm25 = BM25Okapi([lexical_tokens(t) for t in texts])

    corpus_embeddings = {key: np.load(OUT_DIR / cfg["embeddings_file"]) for key, cfg in MODELS.items()}
    encoders = {
        key: SentenceTransformer(cfg["name"], revision=cfg["revision"], device="cpu", cache_folder=str(CACHE_HF))
        for key, cfg in MODELS.items()
    }

    per_query_rows = []
    for query_id in query_ids:
        gold = gold_by_query[query_id]
        text = query_texts[query_id]

        rankings = {"bm25": rank_bm25(bm25, text, ids)}
        for key, cfg in MODELS.items():
            qvec = encoders[key].encode([cfg["query_prefix"] + text], normalize_embeddings=True)[0]
            rankings[key] = rank_dense(corpus_embeddings[key], qvec, ids)

        row = {"query_id": query_id, "query_text": text, "n_gold": len(gold)}
        for method, ranked_ids in rankings.items():
            row[f"{method}_recall@{K}"] = recall_at_k(ranked_ids, gold, K)
            row[f"{method}_ndcg@{K}"] = ndcg_at_k(ranked_ids, gold, K)
        per_query_rows.append(row)

        print(f"\n{query_id}  ({text!r}, {len(gold)} oro)")
        for method in rankings:
            print(f"  {method:<7} recall@{K}={row[f'{method}_recall@{K}']:.2f}  nDCG@{K}={row[f'{method}_ndcg@{K}']:.3f}")

    methods = ["bm25", "minilm", "bge"]
    means = {m: {"recall": np.mean([r[f"{m}_recall@{K}"] for r in per_query_rows]),
                 "ndcg": np.mean([r[f"{m}_ndcg@{K}"] for r in per_query_rows]),
                 "recall_values": [r[f"{m}_recall@{K}"] for r in per_query_rows],
                 "ndcg_values": [r[f"{m}_ndcg@{K}"] for r in per_query_rows]} for m in methods}

    print(f"\n=== Media su {len(query_ids)} domande ===")
    for m in methods:
        print(f"  {m:<7} recall@{K} medio={means[m]['recall']:.3f}   nDCG@{K} medio={means[m]['ndcg']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metriche_12_domande.json").write_text(
        json.dumps({"k": K, "per_query": per_query_rows,
                    "means": {m: {"recall": means[m]["recall"], "ndcg": means[m]["ndcg"]} for m in methods}},
                   indent=2),
        encoding="utf-8",
    )

    # Figura: barre = media sulle 12 domande, punti = ogni singola domanda
    # (leggermente sparpagliati in orizzontale per non sovrapporsi), così la
    # media non nasconde quanto i risultati oscillano da domanda a domanda.
    fig, (ax_recall, ax_ndcg) = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(methods))
    rng = np.random.default_rng(0)

    for ax, metric_key, title in [(ax_recall, "recall_values", f"Recall@{K}"), (ax_ndcg, "ndcg_values", f"nDCG@{K}")]:
        bar_values = [means[m]["recall"] if metric_key == "recall_values" else means[m]["ndcg"] for m in methods]
        ax.bar(x, bar_values, color="lightsteelblue", edgecolor="steelblue", width=0.6, zorder=2)
        for i, m in enumerate(methods):
            values = means[m][metric_key]
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(np.full(len(values), i) + jitter, values, color="black", s=18, alpha=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["BM25", "MiniLM", "BGE-small"])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)

    fig.suptitle(f"Media su {len(query_ids)} domande del pilota (barre) — ogni punto è una singola domanda")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUT_DIR / "metriche_12_domande.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out_path}")


if __name__ == "__main__":
    main()
