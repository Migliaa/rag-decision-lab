"""Passo 6, prima parte: trovare le domande dove il recupero semplice e quello selezionato
producono contesti diversi nei primi dieci.

Il confronto richiesto dal piano (recupero semplice contro selezionato, a parita' di
generatore) non ha senso misurato su tutte le 180 domande: il riordino cambia il contesto solo
dove sposta l'oro dentro o fuori dai primi dieci, sulle altre il contesto è identico e la
generazione non può differire. Qui si isola quel sottoinsieme, prima di generare qualunque
cosa (Appunti1, conseguenza 2).

Semplice: bge-small-en-v1.5 da solo sulla domanda riscritta.
Selezionato: fusione RRF multi-formulazione + reranking cross-encoder sui primi 30 (16_configurazione_finale.py),
già in cache in esperimenti/studio-embedding-retrieval/output/.

"Differire" qui significa: l'insieme dei 10 passaggi non e' lo stesso, non solo l'ordine --
quello che conta per la generazione e' quali informazioni arrivano, non la posizione esatta.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-generazione/04_domande_dove_differisce.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
MTRAG = REPO_ROOT / "data" / "raw" / "mtrag-source" / "mtrag-human" / "retrieval_tasks" / "fiqa"
DEV_QRELS = MTRAG / "qrels" / "dev.tsv"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"
M1_OUT = REPO_ROOT / "esperimenti" / "studio-embedding-retrieval" / "output"
OUT_DIR = Path(__file__).resolve().parent / "output"

RERANK_DEPTH = 30
K = 10
BGE = dict(name="BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
           query_prefix="Represent this sentence for searching relevant passages: ")


def load_corpus(path: Path):
    ids = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["_id"])
    return ids


def load_qrels(path: Path):
    gold = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            q, c, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                gold.setdefault(q, set()).add(c)
    return gold


def load_variant(path: Path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["_id"]] = r["text"].replace("|user|:", " ").strip()
    return out


def main() -> None:
    ids = load_corpus(CORPUS_PATH)
    id_to_idx = {c: i for i, c in enumerate(ids)}
    gold_by_query = load_qrels(DEV_QRELS)
    query_ids = sorted(gold_by_query)
    rewrite = load_variant(MTRAG / "fiqa_rewrite.jsonl")

    # selezionato: riusa la cache di M1, stessa logica di 01_lettura_manuale.py
    d = np.load(M1_OUT / "configurazione_finale_candidati.npz", allow_pickle=True)
    cand = d["cand"]
    sc = np.load(M1_OUT / "configurazione_finale_scores.npy")
    selezionato = []
    for i in range(len(query_ids)):
        testa = list(np.argsort(-sc[i][:RERANK_DEPTH], kind="stable"))
        selezionato.append([int(cand[i][p]) for p in testa[:K]])

    # semplice: bge-small da solo, incorporamento query al volo (180 domande, pochi secondi)
    emb_bge = np.load(M1_OUT / "corpus_embeddings_bge_full.npy")
    model = SentenceTransformer(BGE["name"], revision=BGE["revision"], device="cpu",
                                 cache_folder=str(CACHE_HF))
    qtexts = [BGE["query_prefix"] + rewrite[q] for q in query_ids]
    qv = model.encode(qtexts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    semplice = []
    for i in range(len(query_ids)):
        scores = emb_bge @ qv[i]
        top = np.argsort(-scores, kind="stable")[:K]
        semplice.append([int(j) for j in top])

    differenti = []
    for i, q in enumerate(query_ids):
        set_sel, set_sem = set(selezionato[i]), set(semplice[i])
        if set_sel != set_sem:
            gi = {id_to_idx[g] for g in gold_by_query[q] if g in id_to_idx}
            differenti.append({
                "query_id": q,
                "passaggi_solo_selezionato": len(set_sel - set_sem),
                "passaggi_solo_semplice": len(set_sem - set_sel),
                "sovrapposizione": len(set_sel & set_sem),
                "oro_in_selezionato": len(gi & set_sel),
                "oro_in_semplice": len(gi & set_sem),
            })

    print(f"Domande dove i due contesti differiscono nei primi {K}: {len(differenti)}/{len(query_ids)}")
    cambia_esito_oro = sum(1 for r in differenti if r["oro_in_selezionato"] != r["oro_in_semplice"])
    print(f"  di cui con quantita' di oro diversa fra i due contesti: {cambia_esito_oro}")

    (OUT_DIR / "domande_dove_differisce.json").write_text(
        json.dumps({"totale_domande": len(query_ids), "differenti": differenti}, indent=2),
        encoding="utf-8")
    print(f"scritto: {OUT_DIR / 'domande_dove_differisce.json'}")


if __name__ == "__main__":
    main()
