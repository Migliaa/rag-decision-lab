"""M3, primo esercizio pratico: un agente minimo con ciclo ReAct scritto a mano (non nascosto
dietro il function-calling nativo dell'SDK), per vedere il meccanismo passo per passo.

Riusa il corpus FiQA gia' presente nel progetto, nessun dato nuovo costruito apposta. Due
strumenti soltanto, coerenti con la forma reale del corpus (passaggi di forum, senza cartelle):
  - search_text(query): ricerca lessicale (conteggio di termini in comune), non vettoriale --
    e' deliberatamente un meccanismo diverso da quello di M1, per vedere se l'iterazione recupera
    qualcosa che una ricerca vettoriale singola aveva perso.
  - read(passage_id): legge un passaggio per intero dato il suo id, visto nei risultati di search_text.

Domanda di prova: quella di regime C in Appunti1 (M2), "Why do I still have to pay income tax?",
gia' caratterizzata come fallimento del recupero vettoriale (nessuna fonte annotata nei primi 60
candidati della fusione RRF + reranking). Le fonti annotate sono note qui solo per verificare il
risultato a posteriori, non entrano mai nel prompt.

Limite di iterazioni imposto dal codice (MAX_STEP), non lasciato alla decisione del modello --
il punto del Modulo 2: non ci si fida che un agente si fermi da solo.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-agenti/01_agente_ricerca_testuale.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "fiqa" / "fiqa.jsonl"
ENV_PATH = REPO_ROOT / "esperimenti" / "studio-generazione" / "apikey.env"
OUT_DIR = Path(__file__).resolve().parent / "output"

MODEL = "gemini-3.1-pro-preview"
MAX_STEP = 6
MAX_CHARS_READ = 1500

DOMANDA = "Why do I still have to pay income tax?"
FONTI_ANNOTATE = {"41509-0-965", "280081-0-1578"}  # solo per verifica a posteriori

SYSTEM = """You are a research agent answering a question using a search tool over a forum \
corpus. You do NOT get any passages upfront: you must search for them yourself.

At each turn, respond with EXACTLY one of these two forms, nothing else:

Thought: <your reasoning>
Action: search_text["<query>"]

or

Thought: <your reasoning>
Action: read["<passage_id>"]

or, when you have enough information (or have concluded you cannot find enough):

Thought: <your reasoning>
Final Answer: <your answer, citing passage ids you used, or stating the search did not find enough>

You have at most {max_step} actions before you must give a Final Answer. Do not invent passage \
ids you have not seen in a search result. Do not repeat an identical search query."""


def load_corpus(path: Path):
    ids, titles, texts = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["_id"])
            titles.append(r.get("title", ""))
            texts.append(r["text"])
    return ids, titles, texts


def lexical_tokens(t: str):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def make_tools(ids, titles, texts):
    id_to_idx = {c: i for i, c in enumerate(ids)}
    tokenized = [lexical_tokens((titles[i] + " " + texts[i])) for i in range(len(ids))]

    def search_text(query: str, k: int = 5):
        qtok = lexical_tokens(query)
        scored = [(len(qtok & tokenized[i]), i) for i in range(len(ids))]
        scored.sort(key=lambda x: -x[0])
        top = [i for score, i in scored[:k] if score > 0]
        if not top:
            return "No results."
        righe = []
        for i in top:
            snippet = texts[i][:200].replace("\n", " ")
            righe.append(f'[{ids[i]}] {snippet}...')
        return "\n".join(righe)

    def read(passage_id: str):
        if passage_id not in id_to_idx:
            return f"Unknown passage id: {passage_id}"
        i = id_to_idx[passage_id]
        return texts[i][:MAX_CHARS_READ]

    return search_text, read


ACTION_RE = re.compile(r'Action:\s*(search_text|read)\["(.*?)"\]', re.DOTALL)
FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def main() -> None:
    load_dotenv(ENV_PATH)
    client = genai.Client(api_key=os.environ["GEMINI_API"])

    ids, titles, texts = load_corpus(CORPUS_PATH)
    search_text, read = make_tools(ids, titles, texts)

    transcript = [f"Question: {DOMANDA}"]
    log = []
    final_answer = None
    forced_stop = False

    for step in range(1, MAX_STEP + 1):
        prompt = SYSTEM.format(max_step=MAX_STEP) + "\n\n" + "\n\n".join(transcript)
        resp = client.models.generate_content(model=MODEL, contents=prompt).text.strip()

        final_match = FINAL_RE.search(resp)
        action_match = ACTION_RE.search(resp)

        if final_match:
            final_answer = final_match.group(1).strip()
            log.append({"step": step, "modello_output": resp})
            transcript.append(resp)
            print(f"--- step {step} ---\n{resp}\n")
            break

        if action_match:
            tool, arg = action_match.group(1), action_match.group(2)
            if tool == "search_text":
                osservazione = search_text(arg)
            else:
                osservazione = read(arg)
            transcript.append(resp)
            transcript.append(f"Observation: {osservazione}")
            log.append({"step": step, "modello_output": resp, "tool": tool, "arg": arg,
                        "osservazione": osservazione})
            print(f"--- step {step} ---\n{resp}\nObservation: {osservazione}\n")
        else:
            # risposta fuori formato: registrata, non interpretata come azione
            transcript.append(resp)
            transcript.append("Observation: (unparseable response, please follow the exact format)")
            log.append({"step": step, "modello_output": resp, "tool": None})
            print(f"--- step {step} (fuori formato) ---\n{resp}\n")

    else:
        forced_stop = True
        print(f"[fermato dal codice dopo {MAX_STEP} passi, nessuna Final Answer emessa]")

    passaggi_letti = {e["arg"] for e in log if e.get("tool") == "read"}
    oro_letto = passaggi_letti & FONTI_ANNOTATE

    risultato = {
        "domanda": DOMANDA,
        "max_step": MAX_STEP,
        "forced_stop": forced_stop,
        "final_answer": final_answer,
        "passaggi_letti": sorted(passaggi_letti),
        "fonti_annotate_trovate": sorted(oro_letto),
        "log": log,
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "agente_ricerca_testuale.json").write_text(
        json.dumps(risultato, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFonti annotate trovate dall'agente: {oro_letto or 'nessuna'}")
    print("scritto: output/agente_ricerca_testuale.json")


if __name__ == "__main__":
    main()
