"""Step 16: combinare le due leve che hanno funzionato, e misurare il sistema completo.

Le misure precedenti hanno isolato due guadagni indipendenti:
- interrogare con piu' formulazioni alza la quota di oro presente nella lista
  (step 14: a 50 candidati da 0.687 a 0.700 con riscritta + ultimo turno);
- rerankare pochi candidati con un cross-encoder piccolo converte in alto una quota
  molto maggiore di quell'oro (step 12b: 79% a 30 candidati contro 61% a 100).

Nessuna delle due e' stata provata sopra l'altra. Qui si costruisce la lista con la
configurazione multi-formulazione e la si riordina alle profondita' migliori, per
misurare se i due guadagni si sommano o si sovrappongono.

Confronto finale contro il punto di partenza del progetto: BGE-small da solo.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/16_configurazione_finale.py
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

RRF_C = 60
K = 10
CANDIDATES = 60                 # margine sopra la profondita' migliore osservata (30)
RERANK_DEPTHS = [20, 30, 50, 60]
VARIANTS = ["fiqa_rewrite", "fiqa_lastturn"]   # configurazione migliore a 50 candidati (step 14)
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RNG = np.random.default_rng(0)

MODELS = {
    "minilm": dict(name="sentence-transformers/all-MiniLM-L6-v2",
                   revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                   query_prefix="", embeddings_file="corpus_embeddings_minilm_full.npy"),
    "bge": dict(name="BAAI/bge-small-en-v1.5",
                revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                query_prefix="Represent this sentence for searching relevant passages: ",
                embeddings_file="corpus_embeddings_bge_full.npy"),
}


def indexed_text(p):
    t = p.get("title", "")
    return f"{t}\n{p['text']}" if t else p["text"]


def lexical_tokens(t):
    return re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", t.lower())


def load_corpus(path):
    ids, texts = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["_id"])
            texts.append(indexed_text(r))
    return ids, texts


def load_qrels(path):
    gold = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            q, c, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                gold.setdefault(q, set()).add(c)
    return gold


def load_variant(path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["_id"]] = r["text"].replace("|user|:", " ").strip()
    return out


def ranks_from_scores(s):
    order = np.argsort(-s, kind="stable")
    r = np.empty_like(order)
    r[order] = np.arange(1, len(order) + 1)
    return r


def metrics(final_ranks):
    recall = sum(1 for r in final_ranks if r <= K) / len(final_ranks)
    dcg = sum(1.0 / math.log2(r + 1) for r in final_ranks if r <= K)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(final_ranks), K) + 1))
    return recall, (dcg / idcg if idcg else 0.0)


def ci(vals, samples=2000):
    a = np.asarray(vals)
    idx = RNG.integers(0, len(a), size=(samples, len(a)))
    m = a[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ids, texts = load_corpus(CORPUS_PATH)
    id_to_idx = {c: i for i, c in enumerate(ids)}
    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    variant_texts = {v: load_variant(MTRAG / f"{v}.jsonl") for v in VARIANTS}
    print(f"Domande: {len(query_ids)}")

    cache = OUT_DIR / "configurazione_finale_candidati.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        cand, gold_ranks, bge_ranks = d["cand"], [list(x) for x in d["gold_ranks"]], [list(x) for x in d["bge_ranks"]]
        print("Candidati multi-formulazione gia' calcolati.")
    else:
        print("Indice BM25...")
        bm25 = BM25Okapi([lexical_tokens(t) for t in texts])
        emb = {k: np.load(OUT_DIR / c["embeddings_file"]) for k, c in MODELS.items()}
        enc = {k: SentenceTransformer(c["name"], revision=c["revision"], device="cpu",
                                      cache_folder=str(CACHE_HF)) for k, c in MODELS.items()}
        qv = {}
        for v in VARIANTS:
            for k, c in MODELS.items():
                qv[(v, k)] = enc[k].encode([c["query_prefix"] + variant_texts[v][q] for q in query_ids],
                                           normalize_embeddings=True, batch_size=32, show_progress_bar=False)

        cand = np.zeros((len(query_ids), CANDIDATES), dtype=np.int64)
        gold_ranks, bge_ranks = [], []
        for i, q in enumerate(query_ids):
            gi = [id_to_idx[g] for g in gold_by_query[q] if g in id_to_idx]
            fused = np.zeros(len(ids))
            n_members = len(VARIANTS) * (len(MODELS) + 1)
            for v in VARIANTS:
                fused += (1 / n_members) / (RRF_C + ranks_from_scores(
                    np.asarray(bm25.get_scores(lexical_tokens(variant_texts[v][q])))))
                for k in MODELS:
                    fused += (1 / n_members) / (RRF_C + ranks_from_scores(emb[k] @ qv[(v, k)][i]))
            fr = ranks_from_scores(fused)
            cand[i] = np.argsort(-fused, kind="stable")[:CANDIDATES]
            gold_ranks.append([int(fr[j]) for j in gi])
            # riferimento: BGE-small da solo, sulla domanda riscritta
            br = ranks_from_scores(emb["bge"] @ qv[("fiqa_rewrite", "bge")][i])
            bge_ranks.append([int(br[j]) for j in gi])
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(query_ids)}", flush=True)
        np.savez(cache, cand=cand, gold_ranks=np.array(gold_ranks, dtype=object),
                 bge_ranks=np.array(bge_ranks, dtype=object))

    print(f"\nCarico il reranker {RERANKER}...")
    model = CrossEncoder(RERANKER, device="cpu", cache_folder=str(CACHE_HF), max_length=512)
    scores_path = OUT_DIR / "configurazione_finale_scores.npy"
    prog_path = OUT_DIR / "configurazione_finale_progress.json"
    if scores_path.exists() and prog_path.exists():
        sc = np.load(scores_path)
        done = json.loads(prog_path.read_text())["done"]
    else:
        sc = np.full((len(query_ids), CANDIDATES), np.nan, dtype=np.float32)
        done = 0
    t0 = time.time()
    for i in range(done, len(query_ids)):
        pairs = [(variant_texts["fiqa_rewrite"][query_ids[i]], texts[p]) for p in cand[i]]
        sc[i] = np.asarray(model.predict(pairs, batch_size=32, show_progress_bar=False), dtype=np.float32)
        np.save(scores_path, sc)
        prog_path.write_text(json.dumps({"done": i + 1, "total": len(query_ids)}))
        if (i + 1) % 20 == 0:
            per_q = (time.time() - t0) / (i - done + 1)
            print(f"  rerank {i + 1}/{len(query_ids)} ({per_q:.1f} s/domanda)", flush=True)

    results = {}

    def record(name, per_query):
        rec = float(np.mean([v[0] for v in per_query]))
        nd = float(np.mean([v[1] for v in per_query]))
        lo, hi = ci([v[0] for v in per_query])
        results[name] = {"recall@10": rec, "ndcg@10": nd, "ic95": [lo, hi]}
        print(f"  {name:<52} recall@10={rec:.3f} [{lo:.3f}-{hi:.3f}]  nDCG@10={nd:.3f}")

    print("\n=== Sistema completo contro i riferimenti ===")
    record("BGE-small da solo (punto di partenza del progetto)", [metrics(g) for g in bge_ranks])
    record("fusione multi-formulazione, senza reranking", [metrics(g) for g in gold_ranks])

    ceiling = {d: float(np.mean([sum(1 for r in g if r <= d) / len(g) for g in gold_ranks])) for d in RERANK_DEPTHS}
    for d in RERANK_DEPTHS:
        per_q = []
        for i, g in enumerate(gold_ranks):
            order = np.argsort(-sc[i][:d], kind="stable")
            pos = {int(p): r + 1 for r, p in enumerate(order)}
            per_q.append(metrics([pos[r - 1] if r <= d else r for r in g]))
        record(f"multi-formulazione + reranking sui primi {d} (tetto {ceiling[d]:.3f})", per_q)

    (OUT_DIR / "configurazione_finale.json").write_text(
        json.dumps({"varianti": VARIANTS, "reranker": RERANKER, "tetti": ceiling, "risultati": results}, indent=2),
        encoding="utf-8")

    names = list(results)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(names) + 2))
    y = np.arange(len(names))
    ax.barh(y - 0.2, [results[n]["ndcg@10"] for n in names], height=0.38, color="steelblue", label="nDCG@10")
    ax.barh(y + 0.2, [results[n]["recall@10"] for n in names], height=0.38, color="lightsteelblue", label="Recall@10")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("valore medio su 180 domande")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Sistema completo: multi-formulazione + reranking")
    fig.tight_layout()
    out = OUT_DIR / "configurazione_finale.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out}")


if __name__ == "__main__":
    main()
