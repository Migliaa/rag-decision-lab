"""Step 14: alzare il tetto della lista di candidati interrogando con piu' formulazioni.

Il reranking non puo' superare l'oro che non e' nella lista: a 100 candidati il tetto
misurato e' 0.768. Per alzarlo senza cambiare modello di embedding si puo' interrogare
il corpus piu' volte con formulazioni diverse della stessa domanda e fondere le liste.

MTRAG fornisce gia' tre formulazioni per ogni turno di conversazione, senza bisogno di
generarle con un modello generativo:
- lastturn: solo l'ultimo messaggio dell'utente, spesso ellittico ("enterprise value")
- questions: tutti i turni concatenati, prolisso e rumoroso
- rewrite: la domanda riscritta in forma autosufficiente da un annotatore

Da sole valgono molto diversamente (recall@10 con BGE: 0.356 / 0.196 / 0.417), ma
sbagliano su domande diverse: se l'errore e' poco correlato, la fusione guadagna.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/14_multiquery.py
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
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

RRF_C = 60
DEPTHS = [10, 20, 50, 100, 200, 500]
VARIANTS = ["fiqa_rewrite", "fiqa_lastturn", "fiqa_questions"]

MODELS = {
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2",
                   revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                   query_prefix="", embeddings_file="corpus_embeddings_minilm_full.npy"),
    "bge": dict(name="BAAI/bge-small-en-v1.5",
                revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                query_prefix="Represent this sentence for searching relevant passages: ",
                embeddings_file="corpus_embeddings_bge_full.npy"),
}

# Combinazioni da confrontare: quali (variante, metodo) entrano nella fusione.
COMBOS = {
    "attuale: rewrite x 3 metodi": [(v, m) for v in ["fiqa_rewrite"] for m in ["bm25", "minilm", "bge"]],
    "rewrite + lastturn x 3 metodi": [(v, m) for v in ["fiqa_rewrite", "fiqa_lastturn"] for m in ["bm25", "minilm", "bge"]],
    "3 formulazioni x 3 metodi": [(v, m) for v in VARIANTS for m in ["bm25", "minilm", "bge"]],
    "3 formulazioni, solo BGE": [(v, "bge") for v in VARIANTS],
    "3 formulazioni, solo densi": [(v, m) for v in VARIANTS for m in ["minilm", "bge"]],
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


def load_qrels(path: Path) -> dict[str, set[str]]:
    gold: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            query_id, corpus_id, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                gold.setdefault(query_id, set()).add(corpus_id)
    return gold


def load_variant_texts(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["_id"]] = row["text"].replace("|user|:", " ").strip()
    return out


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    r = np.empty_like(order)
    r[order] = np.arange(1, len(order) + 1)
    return r


def recall_at_k(gold_ranks, k):
    return sum(1 for r in gold_ranks if r <= k) / len(gold_ranks)


def ndcg_at_k(gold_ranks, k):
    dcg = sum(1.0 / math.log2(r + 1) for r in gold_ranks if r <= k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_ranks), k) + 1))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    ids, texts = load_corpus(CORPUS_PATH)
    id_to_idx = {c: i for i, c in enumerate(ids)}
    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    variant_texts = {v: load_variant_texts(MTRAG / f"{v}.jsonl") for v in VARIANTS}
    print(f"Domande: {len(query_ids)}   passaggi: {len(ids)}")

    print("Indice BM25...")
    bm25 = BM25Okapi([lexical_tokens(t) for t in texts])

    corpus_embeddings = {k: np.load(OUT_DIR / cfg["embeddings_file"]) for k, cfg in MODELS.items()}
    encoders = {k: SentenceTransformer(cfg["name"], revision=cfg["revision"], device="cpu",
                                       cache_folder=str(CACHE_HF)) for k, cfg in MODELS.items()}
    qvecs = {}
    for variant in VARIANTS:
        for key, cfg in MODELS.items():
            qvecs[(variant, key)] = encoders[key].encode(
                [cfg["query_prefix"] + variant_texts[variant][q] for q in query_ids],
                normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    per_query = {name: {f"recall@{k}": [] for k in DEPTHS} | {"ndcg@10": []} for name in COMBOS}

    for i, query_id in enumerate(query_ids):
        gold_idx = [id_to_idx[g] for g in gold_by_query[query_id] if g in id_to_idx]
        if not gold_idx:
            continue

        ranks = {}
        for variant in VARIANTS:
            text = variant_texts[variant][query_id]
            ranks[(variant, "bm25")] = ranks_from_scores(np.asarray(bm25.get_scores(lexical_tokens(text))))
            for key in MODELS:
                ranks[(variant, key)] = ranks_from_scores(corpus_embeddings[key] @ qvecs[(variant, key)][i])

        for name, members in COMBOS.items():
            fused = np.zeros(len(ids), dtype=np.float64)
            for member in members:
                fused += (1.0 / len(members)) * (1.0 / (RRF_C + ranks[member]))
            fr = ranks_from_scores(fused)
            gr = [int(fr[j]) for j in gold_idx]
            for k in DEPTHS:
                per_query[name][f"recall@{k}"].append(recall_at_k(gr, k))
            per_query[name]["ndcg@10"].append(ndcg_at_k(gr, 10))

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(query_ids)}", flush=True)

    summary = {name: {m: float(np.mean(v)) for m, v in d.items()} for name, d in per_query.items()}

    print("\n=== Tetto della lista di candidati, per combinazione di formulazioni ===")
    header = "  " + " " * 34 + "".join(f"@{k:<7}" for k in DEPTHS) + "nDCG@10"
    print(header)
    for name, vals in summary.items():
        row = "".join(f"{vals[f'recall@{k}']:.3f}   " for k in DEPTHS)
        print(f"  {name:<34}{row}{vals['ndcg@10']:.3f}")

    (OUT_DIR / "multiquery.json").write_text(json.dumps({"combos": {k: [list(m) for m in v] for k, v in COMBOS.items()},
                                                         "summary": summary}, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, vals in summary.items():
        ax.plot(DEPTHS, [vals[f"recall@{k}"] for k in DEPTHS], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xticks(DEPTHS)
    ax.set_xticklabels([str(k) for k in DEPTHS])
    ax.set_xlabel("profondita' della lista di candidati")
    ax.set_ylabel("quota di oro presente nella lista (180 domande)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Interrogare con piu' formulazioni alza il tetto del reranking")
    fig.tight_layout()
    out_path = OUT_DIR / "multiquery.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out_path}")


if __name__ == "__main__":
    main()
