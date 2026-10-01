"""Step 5: due modi di misurare la stessa classifica, sulla domanda di 03/04.

Recall@k e nDCG@k rispondono a due domande diverse sulla STESSA classifica:

- Recall@k: quota di passaggi oro che compaiono da qualche parte nei primi k.
  Non guarda la posizione: un oro al 1° posto e uno al 9° posto in top-10
  contano allo stesso modo.
- nDCG@k (normalized Discounted Cumulative Gain): premia l'oro trovato in
  alto nella classifica più di quello trovato in fondo. Un modello che mette
  tutto l'oro ai primi posti ha nDCG più alto di uno che lo trova ma in
  fondo alla lista, anche a parità di recall.

Su questa domanda i due modelli hanno lo STESSO recall@10 (trovano tutto
l'oro) ma classifiche diverse: qui nDCG deve differenziarli, recall no.
Un solo esempio: non generalizzare il confronto da qui (step successivo:
farlo su tutte le 12 domande del pilota).

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/05_metriche_singola_query.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "output"


def recall_at_k(ranking: list[dict], gold_ids: set[str], k: int) -> float:
    top_k_ids = {row["id"] for row in ranking[:k]}
    found = top_k_ids & gold_ids
    return len(found) / len(gold_ids)


def dcg_at_k(ranking: list[dict], gold_ids: set[str], k: int) -> float:
    """Somma 1/log2(rank+1) per ogni passaggio oro trovato, rank a partire da 1.

    Rilevanza binaria qui (le qrels del pilota usano solo score=1): un
    passaggio o è rilevante (contributo 1/log2(rank+1)) o non lo è
    (contributo 0). Con rilevanza graduata (0,1,2,...) il numeratore
    sarebbe 2^rel - 1 invece di rel: qui coincidono perché rel è sempre 0 o 1.
    """
    total = 0.0
    for row in ranking[:k]:
        if row["id"] in gold_ids:
            total += 1.0 / math.log2(row["rank"] + 1)
    return total


def idcg_at_k(n_gold: int, k: int) -> float:
    """Il DCG della classifica ideale: tutto l'oro compattato in cima.

    Serve come denominatore per normalizzare: senza dividere per questo,
    un numero di oro diverso tra due domande renderebbe i DCG grezzi non
    confrontabili (una domanda con 5 oro ha più margine di punteggio di una
    con 2, a prescindere da quanto bene ha lavorato il retriever).
    """
    n_at_top = min(n_gold, k)
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, n_at_top + 1))


def ndcg_at_k(ranking: list[dict], gold_ids: set[str], k: int) -> float:
    idcg = idcg_at_k(len(gold_ids), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranking, gold_ids, k) / idcg


def main() -> None:
    results = json.loads((OUT_DIR / "query_ranking_example.json").read_text(encoding="utf-8"))
    gold_ids = set(results["gold_ids"])
    k = 10

    print(f"Domanda: {results['query_text']!r}")
    print(f"Oro noto: {len(gold_ids)} passaggi\n")

    metrics = {}
    for model_key, model_results in results["models"].items():
        ranking = model_results["ranking"]
        r = recall_at_k(ranking, gold_ids, k)
        n = ndcg_at_k(ranking, gold_ids, k)
        ranks_of_gold = sorted(row["rank"] for row in ranking if row["id"] in gold_ids)

        print(f"--- {model_key} ---")
        print(f"  Posizioni dell'oro in classifica: {ranks_of_gold}")
        print(f"  Recall@{k}: {r:.3f}")
        print(f"  nDCG@{k}:   {n:.3f}")
        print()

        metrics[model_key] = {"recall_at_k": r, "ndcg_at_k": n, "gold_ranks": ranks_of_gold}

    out_path = OUT_DIR / "metriche_singola_query.json"
    out_path.write_text(json.dumps({"query_text": results["query_text"], "k": k, "models": metrics}, indent=2), encoding="utf-8")
    print(f"Metriche salvate in: {out_path}")


if __name__ == "__main__":
    main()
