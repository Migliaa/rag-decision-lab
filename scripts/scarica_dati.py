"""Scarica da zero i dati MTRAG/FiQA usati nel progetto e verifica gli hash del manifest.

Prende dal repository pubblico IBM `mt-rag-benchmark`, alla revisione congelata, soltanto il corpus
FiQA a livello di passaggio, le domande e le qrels (una copia Git parziale, pochi MB). Il corpus
viene estratto in `data/raw/fiqa/fiqa.jsonl`. Gli hash attesi sono quelli di `data/manifest.json`.

Uso, dalla radice del repository:
    uv run python scripts/scarica_dati.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SORGENTE = ROOT / "data/raw/mtrag-source"
REPO = "https://github.com/IBM/mt-rag-benchmark.git"
PERCORSI = ["corpora/passage_level", "mtrag-human/retrieval_tasks/fiqa"]
MANIFEST = ROOT / "data/manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blocco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blocco)
    return h.hexdigest()


def git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    revisione = manifest["source_revision"]
    atteso = manifest["inputs"]

    if not SORGENTE.exists():
        SORGENTE.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--filter=blob:none", "--sparse", "--no-checkout", REPO, str(SORGENTE))
    git("sparse-checkout", "set", *PERCORSI, cwd=SORGENTE)
    git("checkout", revisione, cwd=SORGENTE)

    archivio = SORGENTE / "corpora/passage_level/fiqa.jsonl.zip"
    destinazione = ROOT / "data/raw/fiqa/fiqa.jsonl"
    destinazione.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archivio) as z:
        nome = next(n for n in z.namelist() if n.endswith(".jsonl"))
        destinazione.write_bytes(z.read(nome))

    domande = SORGENTE / "mtrag-human/retrieval_tasks/fiqa/fiqa_lastturn.jsonl"
    qrels = SORGENTE / "mtrag-human/retrieval_tasks/fiqa/qrels/dev.tsv"
    controlli = {
        "corpus_zip_sha256": archivio,
        "corpus_jsonl_sha256": destinazione,
        "queries_sha256": domande,
        "qrels_sha256": qrels,
    }
    errori = 0
    for chiave, percorso in controlli.items():
        trovato = sha256(percorso)
        ok = trovato == atteso[chiave]
        errori += not ok
        print(f"{'ok ' if ok else 'DIVERSO'} {chiave}")
    if errori:
        sys.exit("Gli hash non coincidono con data/manifest.json: dati diversi da quelli del progetto.")
    print("Dati scaricati e verificati.")


if __name__ == "__main__":
    main()
