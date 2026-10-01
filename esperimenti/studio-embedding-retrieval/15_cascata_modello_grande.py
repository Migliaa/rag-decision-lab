"""Step 15: cascata vera - il reranker grande applicato solo ai pochi sopravvissuti.

bge-reranker-large costa 2,7 minuti per domanda su 100 candidati: otto ore per una
valutazione, fuori budget (step 11). Ma il punto della cascata e' proprio questo: se un
modello economico riduce prima i candidati da 100 a 20, il modello grande ne vede un
quinto e il costo scende nella stessa proporzione, da otto ore a circa un'ora e mezza.

Misura quindi se il modello grande, dove e' possibile permetterselo, aggiunge qualcosa
sopra il modello piccolo - cioe' se la cascata vale il suo costo o se tanto vale
fermarsi al modello piccolo.

Ordine finale: i 20 sopravvissuti riordinati da bge-reranker-large occupano le prime 20
posizioni, il resto della lista conserva l'ordine dato dal reranker economico.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/15_cascata_modello_grande.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
OUT_DIR = Path(__file__).resolve().parent / "output"

CHEAP_KEY = "ms-marco-MiniLM-L6"
CHEAP_DEPTH = 100      # quanti candidati vede il modello economico
SURVIVORS = 20         # quanti ne passano al modello grande
K = 10

BIG_NAME = "BAAI/bge-reranker-large"
BIG_REVISION = "55611d7bca2a7133960a6d3b71e083071bbfc312"


def indexed_text(p: dict) -> str:
    t = p.get("title", "")
    return f"{t}\n{p['text']}" if t else p["text"]


def load_corpus(path: Path):
    ids, texts = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["_id"])
            texts.append(indexed_text(r))
    return ids, texts


def load_qrels(path: Path):
    gold = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            q, c, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                gold.setdefault(q, set()).add(c)
    return gold


def load_query_texts(path: Path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["_id"]] = r["text"].replace("|user|:", "").strip()
    return out


def metrics(final_ranks):
    recall = sum(1 for r in final_ranks if r <= K) / len(final_ranks)
    dcg = sum(1.0 / math.log2(r + 1) for r in final_ranks if r <= K)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(final_ranks), K) + 1))
    return recall, (dcg / idcg if idcg else 0.0)


def main() -> None:
    ids, texts = load_corpus(CORPUS_PATH)
    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    query_texts = load_query_texts(MTRAG / "fiqa_rewrite.jsonl")

    data = np.load(OUT_DIR / "candidati_180.npz", allow_pickle=True)
    candidates = data["candidates"]
    golds = [list(g) for g in data["gold_ranks_full"]]
    cheap_scores = np.load(OUT_DIR / f"rerank_scores_{CHEAP_KEY}.npy")

    scores_path = OUT_DIR / "cascata_big_scores.npy"
    progress_path = OUT_DIR / "cascata_big_progress.json"
    if scores_path.exists() and progress_path.exists():
        big = np.load(scores_path)
        done = json.loads(progress_path.read_text())["done"]
        print(f"Riprendo da {done}/{len(query_ids)}")
    else:
        big = np.full((len(query_ids), SURVIVORS), np.nan, dtype=np.float32)
        done = 0

    if done < len(query_ids):
        print(f"Carico {BIG_NAME}...")
        model = CrossEncoder(BIG_NAME, revision=BIG_REVISION, device="cpu",
                             cache_folder=str(CACHE_HF), max_length=512)
        t0 = time.time()
        for i in range(done, len(query_ids)):
            order = np.argsort(-cheap_scores[i][:CHEAP_DEPTH], kind="stable")[:SURVIVORS]
            pairs = [(query_texts[query_ids[i]], texts[candidates[i][p]]) for p in order]
            big[i] = np.asarray(model.predict(pairs, batch_size=8, show_progress_bar=False), dtype=np.float32)
            np.save(scores_path, big)
            progress_path.write_text(json.dumps({"done": i + 1, "total": len(query_ids)}))
            if (i + 1) % 5 == 0 or i == done:
                per_q = (time.time() - t0) / (i - done + 1)
                print(f"  {i + 1}/{len(query_ids)}  ({per_q:.1f} s/domanda, "
                      f"~{per_q * (len(query_ids) - i - 1) / 60:.0f} min rimanenti)", flush=True)

    # valutazione: cascata contro solo modello economico, stessa lista di partenza
    res = {"solo economico": [], "cascata con modello grande": []}
    for i in range(len(query_ids)):
        cheap_order = np.argsort(-cheap_scores[i][:CHEAP_DEPTH], kind="stable")
        surv = cheap_order[:SURVIVORS]
        rest = cheap_order[SURVIVORS:]

        pos_cheap = {int(p): r + 1 for r, p in enumerate(cheap_order)}
        reordered = surv[np.argsort(-big[i], kind="stable")]
        pos_casc = {int(p): r + 1 for r, p in enumerate(np.concatenate([reordered, rest]))}

        for name, pos in (("solo economico", pos_cheap), ("cascata con modello grande", pos_casc)):
            final = [pos[r - 1] if r <= CHEAP_DEPTH else r for r in golds[i]]
            res[name].append(metrics(final))

    print("\n=== Cascata: il modello grande sui primi 20 aggiunge qualcosa? ===")
    summary = {}
    for name, vals in res.items():
        rec = float(np.mean([v[0] for v in vals]))
        nd = float(np.mean([v[1] for v in vals]))
        summary[name] = {"recall@10": rec, "ndcg@10": nd}
        print(f"  {name:<32} recall@10={rec:.3f}  nDCG@10={nd:.3f}")

    (OUT_DIR / "cascata_modello_grande.json").write_text(
        json.dumps({"cheap": CHEAP_KEY, "cheap_depth": CHEAP_DEPTH, "survivors": SURVIVORS,
                    "big": BIG_NAME, "summary": summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
