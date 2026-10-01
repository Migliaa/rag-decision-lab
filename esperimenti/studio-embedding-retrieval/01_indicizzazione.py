"""Step 1: indicizzare il corpus.

Cosa vuol dire "indicizzare" qui: trasformare ogni passaggio del corpus in un
vettore, UNA VOLTA SOLA, e tenere tutti i vettori insieme in una matrice.
Le domande verranno dopo (step 2): questo script non cerca niente, prepara
solo il materiale su cui la ricerca lavorerà.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/01_indicizzazione.py
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sentence_transformers import SentenceTransformer

# Percorsi. CORPUS_PATH punta al sottoinsieme pilota (512 passaggi) già
# preparato in una sessione precedente: stesso schema id/text/title visto
# nella conversazione, solo più piccolo per essere veloce da eseguire qui.
REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "E001-pilot-512" / "corpus.jsonl"
CACHE_HF = REPO_ROOT / "data" / "cache" / "huggingface"  # modello già scaricato qui
OUT_DIR = Path(__file__).resolve().parent / "output"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # fissata: non cambia da sola


def load_corpus(path: Path) -> list[dict]:
    """Legge il corpus riga per riga.

    Ogni riga del file è un oggetto JSON indipendente (formato JSONL, non un
    unico array JSON): un passaggio con id stabile, testo, e titolo (qui
    sempre vuoto per FiQA). L'ordine con cui le righe finiscono in questa
    lista è anche l'ordine delle righe nella matrice di vettori più sotto:
    passages[i] corrisponde a embeddings[i].
    """
    passages = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages


def indexed_text(passage: dict) -> str:
    """Decide che cosa viene effettivamente incorporato.

    Non incorporiamo l'intero oggetto JSON (id compreso): solo il testo che
    un lettore umano leggerebbe. Se un giorno ci fosse un titolo non vuoto,
    lo mettiamo davanti al testo perché fa parte del significato del
    passaggio (stessa scelta già fatta in src/retrieval.py, per coerenza).
    """
    title = passage.get("title", "")
    text = passage["text"]
    return f"{title}\n{text}" if title else text


def main() -> None:
    passages = load_corpus(CORPUS_PATH)
    print(f"Passaggi caricati: {len(passages)}")
    print("Primo passaggio (id):", passages[0]["_id"])
    print("Primo passaggio (primi 120 caratteri di testo):")
    print(indexed_text(passages[0])[:120].replace("\n", " "))

    # Il modello traduce ogni passaggio in un vettore di 384 numeri.
    # device="cpu" perché non abbiamo GPU (vedi vincoli in CONTESTO_STUDIO_LIBERO.md).
    print("\nCarico il modello (da cache locale, nessun download)...")
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
        cache_folder=str(CACHE_HF),
    )

    texts = [indexed_text(p) for p in passages]

    print(f"\nIncorporo {len(texts)} passaggi (batch da 16, una tantum)...")
    started = perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=16,
        # normalize_embeddings=True porta ogni vettore a lunghezza 1.
        # Serve per lo step successivo: con vettori normalizzati, il prodotto
        # scalare tra due vettori equivale alla similarità coseno, cioè
        # "quanto puntano nella stessa direzione", indipendentemente da
        # quanto sono lunghi i testi di partenza.
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = perf_counter() - started

    print(f"\nFatto in {elapsed:.1f} secondi.")
    print(f"Forma della matrice risultante: {embeddings.shape}")
    print(f"  -> {embeddings.shape[0]} righe: un vettore per passaggio, stesso ordine di 'passages'")
    print(f"  -> {embeddings.shape[1]} colonne: dimensioni del vettore, decise dal modello, non da noi")

    # Verifica di sanità: se normalize_embeddings ha funzionato, la lunghezza
    # (norma euclidea) di ogni vettore deve essere 1.0, non qualcos'altro.
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nControllo: norma media dei vettori = {norms.mean():.4f} (atteso: 1.0)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "corpus_embeddings_minilm.npy"
    np.save(out_path, embeddings)
    print(f"\nMatrice salvata in: {out_path}")
    print("Salvo anche gli id nello stesso ordine, per poterli riallineare dopo.")
    ids_path = OUT_DIR / "corpus_ids.json"
    ids_path.write_text(json.dumps([p["_id"] for p in passages]), encoding="utf-8")
    print(f"Id salvati in: {ids_path}")


if __name__ == "__main__":
    main()
