"""Step 7: le stesse 12 domande, ma contro il corpus FiQA COMPLETO (61.022 passaggi)
invece del pilota da 512.

Stesse 12 domande e stesso oro esatto di 06_metriche_12_domande.py: cambia solo la
"scala del pagliaio" in cui cercare. Il pilota conteneva quasi solo distrattori
scelti apposta a riempire un sottoinsieme piccolo; qui i distrattori sono TUTTI
gli altri passaggi reali del corpus FiQA, ~120 volte di più. Se i numeri
peggiorano, non è un errore del codice: è la differenza tra un pilota di
sviluppo e la scala reale.

Tempo atteso: indicizzare 61.022 passaggi con due modelli su CPU richiede
circa un'ora e mezza in totale (proporzionale ai tempi già misurati sui 512
del pilota). Eseguito in background.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/07_metriche_corpus_completo.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"  # corpus completo, non il pilota
PILOT_QRELS_PATH = REPO_ROOT / "data" / "processed" / "E001-pilot-512" / "qrels.tsv"
REWRITE_PATH = (
    REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa" / "fiqa_rewrite.jsonl"
)
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

K = 10
BATCH_SIZE = 64  # più grande che nel pilota: su 61k passaggi conta la resa, non la RAM (24 GB liberi)

MODELS = {
    "minilm": dict(
        name="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        query_prefix="",
        embeddings_file="corpus_embeddings_minilm_full.npy",
    ),
    "bge": dict(
        name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        query_prefix="Represent this sentence for searching relevant passages: ",
        embeddings_file="corpus_embeddings_bge_full.npy",
    ),
}


def indexed_text(passage: dict) -> str:
    title = passage.get("title", "")
    text = passage["text"]
    return f"{title}\n{text}" if title else text


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower())


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["_id"])
            texts.append(indexed_text(row))
    return ids, texts


def load_pilot_query_gold() -> dict[str, set[str]]:
    """Stesse 12 domande e stesso oro esatto del pilota: qui cambia solo il
    corpus in cui cercarle, non le domande né la verità nota."""
    gold_by_query: dict[str, set[str]] = {}
    with PILOT_QRELS_PATH.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            query_id, corpus_id, _score = line.rstrip("\n").split("\t")
            gold_by_query.setdefault(query_id, set()).add(corpus_id)
    return gold_by_query


def load_rewrite_texts(query_ids: set[str], path: Path) -> dict[str, str]:
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


CHUNK_SIZE = 2000  # checkpoint ogni ~2-4 minuti: se il PC si spegne, si perde al massimo un pezzo


def get_or_build_embeddings(model_key: str, texts: list[str]) -> np.ndarray:
    """Come prima, ma ripartibile: salva un checkpoint ogni CHUNK_SIZE passaggi
    (embeddings parziali + quanti ne sono già fatti) invece di salvare solo a
    fine lavoro. Un'interruzione (PC spento, batteria scarica) perde al massimo
    l'ultimo pezzo incompleto, non l'intero lavoro fatto fino a quel momento.
    Rilanciare lo script più tardi riprende da dove si era fermato: non serve
    fare nulla di manuale."""
    cfg = MODELS[model_key]
    cache_path = OUT_DIR / cfg["embeddings_file"]
    progress_path = OUT_DIR / f"{model_key}_full_progress.json"

    if cache_path.exists() and not progress_path.exists():
        print(f"[{model_key}] embedding completi già salvati, li ricarico da {cache_path.name}")
        return np.load(cache_path)

    n_total = len(texts)
    embedding_dim = None
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start_index = progress["completed"]
        embeddings = np.load(cache_path)
        print(f"[{model_key}] riprendo da {start_index}/{n_total} passaggi già incorporati")
    else:
        start_index = 0
        embeddings = None  # allocato al primo chunk, quando conosciamo la dimensione del vettore

    if start_index >= n_total:
        return embeddings

    print(f"[{model_key}] carico il modello...")
    model = SentenceTransformer(cfg["name"], revision=cfg["revision"], device="cpu", cache_folder=str(CACHE_HF))

    started = perf_counter()
    for chunk_start in range(start_index, n_total, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_total)
        chunk_embeddings = model.encode(
            texts[chunk_start:chunk_end], batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False,
        )
        if embeddings is None:
            embedding_dim = chunk_embeddings.shape[1]
            embeddings = np.zeros((n_total, embedding_dim), dtype=chunk_embeddings.dtype)
        embeddings[chunk_start:chunk_end] = chunk_embeddings

        np.save(cache_path, embeddings)
        progress_path.write_text(json.dumps({"completed": chunk_end, "total": n_total}), encoding="utf-8")

        elapsed = perf_counter() - started
        done_now = chunk_end - start_index
        rate = done_now / elapsed if elapsed > 0 else 0
        remaining = (n_total - chunk_end) / rate if rate > 0 else float("nan")
        print(f"[{model_key}] {chunk_end}/{n_total} passaggi "
              f"({elapsed / 60:.1f} min finora, stima rimanente {remaining / 60:.1f} min)")

    progress_path.unlink(missing_ok=True)  # completo: non serve più il checkpoint
    print(f"[{model_key}] fatto in {(perf_counter() - started) / 60:.1f} minuti")
    return embeddings


def main() -> None:
    print("Carico il corpus FiQA completo...")
    ids, texts = load_corpus(CORPUS_PATH)
    print(f"Passaggi nel corpus completo: {len(ids)}")

    gold_by_query = load_pilot_query_gold()
    query_ids = sorted(gold_by_query)
    query_texts = load_rewrite_texts(set(query_ids), REWRITE_PATH)

    print("\nCostruisco l'indice BM25 sul corpus completo...")
    started = perf_counter()
    bm25 = BM25Okapi([lexical_tokens(t) for t in texts])
    print(f"Indice BM25 pronto in {perf_counter() - started:.1f} secondi")

    corpus_embeddings = {key: get_or_build_embeddings(key, texts) for key in MODELS}

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
    means = {m: {"recall": float(np.mean([r[f"{m}_recall@{K}"] for r in per_query_rows])),
                 "ndcg": float(np.mean([r[f"{m}_ndcg@{K}"] for r in per_query_rows])),
                 "recall_values": [r[f"{m}_recall@{K}"] for r in per_query_rows],
                 "ndcg_values": [r[f"{m}_ndcg@{K}"] for r in per_query_rows]} for m in methods}

    print(f"\n=== Media su {len(query_ids)} domande, corpus completo ({len(ids)} passaggi) ===")
    for m in methods:
        print(f"  {m:<7} recall@{K} medio={means[m]['recall']:.3f}   nDCG@{K} medio={means[m]['ndcg']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metriche_corpus_completo.json").write_text(
        json.dumps({"k": K, "n_passages": len(ids), "per_query": per_query_rows,
                    "means": {m: {"recall": means[m]["recall"], "ndcg": means[m]["ndcg"]} for m in methods}},
                   indent=2),
        encoding="utf-8",
    )

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

    fig.suptitle(f"Media su {len(query_ids)} domande, corpus FiQA completo ({len(ids)} passaggi) — barre = media, punti = ogni domanda")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUT_DIR / "metriche_corpus_completo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out_path}")


if __name__ == "__main__":
    main()
