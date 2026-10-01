"""Step 17: lo stesso modello usato per SCEGLIERE i candidati o per RIORDINARLI.

E' la domanda posta da Andrea: invece di incorporare tutti i 61.022 passaggi con un
modello piu' forte, non basterebbe incorporare solo i 100 candidati gia' selezionati e
riordinarli? Costerebbe molto meno.

Qui la si misura invece di ragionarci, perche' avendo l'intero corpus incorporato con
gte-base si possono confrontare le due strade a parita' di modello:

- **scegliere**: gte-base cerca su tutti i 61.022 passaggi e produce lui la lista;
- **riordinare**: la lista resta quella scelta da bge-small e gte-base la riordina soltanto.

La differenza fra le due misura esattamente quanto vale la fase di recupero rispetto alla
fase di riordino. In piu' si confronta con il cross-encoder, che e' l'alternativa reale a
parita' di costo per domanda.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/17_gte_scegliere_o_riordinare.py
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
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

RRF_C = 60
K = 10
DEPTHS = [10, 20, 30, 50, 100, 200]
RERANK_DEPTH = 50
RNG = np.random.default_rng(0)

GTE = dict(name="Alibaba-NLP/gte-base-en-v1.5", file="corpus_embeddings_gte-base_full.npy", prefix="")
BGE = dict(name="BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
           file="corpus_embeddings_bge_full.npy",
           prefix="Represent this sentence for searching relevant passages: ")
MINILM = dict(name="sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
              file="corpus_embeddings_minilm_full.npy", prefix="")
CROSS = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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


def load_queries(path):
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
    return tuple(float(np.percentile(a[idx].mean(axis=1), p)) for p in (2.5, 97.5))


def main() -> None:
    gte_path = OUT_DIR / GTE["file"]
    prog = OUT_DIR / "gte-base_full_progress.json"
    if not gte_path.exists() or json.loads(prog.read_text())["completed"] < 61022:
        done = json.loads(prog.read_text())["completed"] if prog.exists() else 0
        print(f"Embedding gte-base incompleto ({done}/61022). Attendere la fine di 13_embedding_modello_forte.py.")
        return

    ids, texts = load_corpus(CORPUS_PATH)
    id_to_idx = {c: i for i, c in enumerate(ids)}
    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    qtext = load_queries(MTRAG / "fiqa_rewrite.jsonl")
    print(f"Domande: {len(query_ids)}   passaggi: {len(ids)}")

    emb = {"gte": np.load(gte_path), "bge": np.load(OUT_DIR / BGE["file"]),
           "minilm": np.load(OUT_DIR / MINILM["file"])}
    encoders = {
        "gte": SentenceTransformer(GTE["name"], device="cpu", cache_folder=str(CACHE_HF), trust_remote_code=True),
        "bge": SentenceTransformer(BGE["name"], revision=BGE["revision"], device="cpu", cache_folder=str(CACHE_HF)),
        "minilm": SentenceTransformer(MINILM["name"], revision=MINILM["revision"], device="cpu",
                                      cache_folder=str(CACHE_HF)),
    }
    encoders["gte"].max_seq_length = 512
    prefixes = {"gte": GTE["prefix"], "bge": BGE["prefix"], "minilm": MINILM["prefix"]}
    qvec = {k: encoders[k].encode([prefixes[k] + qtext[q] for q in query_ids],
                                  normalize_embeddings=True, batch_size=32, show_progress_bar=False)
            for k in emb}

    print("Indice BM25...")
    bm25 = BM25Okapi([lexical_tokens(t) for t in texts])
    cross = CrossEncoder(CROSS, device="cpu", cache_folder=str(CACHE_HF), max_length=512)

    rows = {name: [] for name in [
        "bge-small da solo",
        "gte-base da solo",
        "fusione con bge-small (senza gte)",
        "fusione con gte-base al posto di bge-small",
        "candidati di bge-small, riordinati da gte-base (l'idea da verificare)",
        "candidati di bge-small, riordinati dal cross-encoder",
        "candidati di gte-base, riordinati dal cross-encoder",
    ]}
    ceilings = {d: [] for d in DEPTHS}
    ceilings_gte = {d: [] for d in DEPTHS}

    for i, q in enumerate(query_ids):
        gi = [id_to_idx[g] for g in gold_by_query[q] if g in id_to_idx]
        if not gi:
            continue
        r_bm25 = ranks_from_scores(np.asarray(bm25.get_scores(lexical_tokens(qtext[q]))))
        r = {k: ranks_from_scores(emb[k] @ qvec[k][i]) for k in emb}

        rows["bge-small da solo"].append(metrics([int(r["bge"][j]) for j in gi]))
        rows["gte-base da solo"].append(metrics([int(r["gte"][j]) for j in gi]))

        def fuse(members):
            tot = np.zeros(len(ids))
            for arr in members:
                tot += (1 / len(members)) / (RRF_C + arr)
            return tot

        f_old = fuse([r_bm25, r["bge"], r["minilm"]])
        f_new = fuse([r_bm25, r["gte"], r["minilm"]])
        rr_old, rr_new = ranks_from_scores(f_old), ranks_from_scores(f_new)
        rows["fusione con bge-small (senza gte)"].append(metrics([int(rr_old[j]) for j in gi]))
        rows["fusione con gte-base al posto di bge-small"].append(metrics([int(rr_new[j]) for j in gi]))

        for d in DEPTHS:
            ceilings[d].append(sum(1 for j in gi if rr_old[j] <= d) / len(gi))
            ceilings_gte[d].append(sum(1 for j in gi if rr_new[j] <= d) / len(gi))

        # la lista da riordinare: i primi RERANK_DEPTH della fusione costruita con bge-small
        cand_old = np.argsort(-f_old, kind="stable")[:RERANK_DEPTH]
        cand_new = np.argsort(-f_new, kind="stable")[:RERANK_DEPTH]

        def reordered_metrics(cand, base_ranks, scores):
            order = cand[np.argsort(-scores, kind="stable")]
            pos = {int(p): k + 1 for k, p in enumerate(order)}
            return metrics([pos[j] if j in pos else int(base_ranks[j]) for j in gi])

        # 1. riordino con gte-base usato come bi-encoder sui candidati (l'idea proposta)
        rows["candidati di bge-small, riordinati da gte-base (l'idea da verificare)"].append(
            reordered_metrics(cand_old, rr_old, emb["gte"][cand_old] @ qvec["gte"][i]))
        # 2. riordino con il cross-encoder, stessa lista
        cs_old = np.asarray(cross.predict([(qtext[q], texts[p]) for p in cand_old], batch_size=32,
                                          show_progress_bar=False))
        rows["candidati di bge-small, riordinati dal cross-encoder"].append(
            reordered_metrics(cand_old, rr_old, cs_old))
        # 3. lista scelta da gte-base, riordino con il cross-encoder
        cs_new = np.asarray(cross.predict([(qtext[q], texts[p]) for p in cand_new], batch_size=32,
                                          show_progress_bar=False))
        rows["candidati di gte-base, riordinati dal cross-encoder"].append(
            reordered_metrics(cand_new, rr_new, cs_new))

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(query_ids)}", flush=True)

    print("\n=== Tetto della lista di candidati ===")
    for d in DEPTHS:
        print(f"  primi {d:3d}: con bge-small {np.mean(ceilings[d]):.3f}   con gte-base {np.mean(ceilings_gte[d]):.3f}")

    print("\n=== Risultati (180 domande) ===")
    summary = {}
    for name, vals in rows.items():
        rec = float(np.mean([v[0] for v in vals]))
        nd = float(np.mean([v[1] for v in vals]))
        lo, hi = ci([v[0] for v in vals])
        summary[name] = {"recall@10": rec, "ndcg@10": nd, "ic95": [lo, hi]}
        print(f"  {name:<62} recall@10={rec:.3f} [{lo:.3f}-{hi:.3f}]  nDCG@10={nd:.3f}")

    (OUT_DIR / "gte_scegliere_o_riordinare.json").write_text(
        json.dumps({"rerank_depth": RERANK_DEPTH,
                    "tetti_bge": {str(d): float(np.mean(ceilings[d])) for d in DEPTHS},
                    "tetti_gte": {str(d): float(np.mean(ceilings_gte[d])) for d in DEPTHS},
                    "risultati": summary}, indent=2), encoding="utf-8")

    names = list(summary)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(names) + 2))
    y = np.arange(len(names))
    ax.barh(y - 0.2, [summary[n]["ndcg@10"] for n in names], height=0.38, color="steelblue", label="nDCG@10")
    ax.barh(y + 0.2, [summary[n]["recall@10"] for n in names], height=0.38, color="lightsteelblue", label="Recall@10")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("valore medio su 180 domande")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Un modello piu' forte: sceglie i candidati o li riordina?")
    fig.tight_layout()
    out = OUT_DIR / "gte_scegliere_o_riordinare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigura salvata in: {out}")


if __name__ == "__main__":
    main()
