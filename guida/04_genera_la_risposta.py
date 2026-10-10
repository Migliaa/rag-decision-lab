"""Esempio 4 — Dalla ricerca alla risposta: il prompt che trasforma i brani in un testo citato.

Cosa fa: cerca i 10 brani migliori per una domanda (come nell'esempio 3), li numera,
li incolla in un prompt insieme alla domanda e, SOLO se lo chiedi, manda il prompt a
un modello generativo (Gemini). Senza l'opzione --chiama-api si limita a stampare il
prompt: nessuna rete, nessun costo.

Il punto di questa tappa è il prompt, non la chiamata: il modello vede soltanto i
brani che gli passiamo. Se la ricerca non porta la fonte giusta fra i 10, il modello
non ha modo di rispondere bene, quindi la qualità della risposta dipende dalla ricerca.

Come si lancia (dalla cartella "rag"):
    uv run python guida/04_genera_la_risposta.py
    uv run --extra generazione python guida/04_genera_la_risposta.py --chiama-api
        (richiede esperimenti/studio-generazione/apikey.env con GEMINI_API=<chiave>
         e può comportare un costo sul tuo account Gemini)

Corrisponde a: esperimenti/studio-generazione/05_confronto_semplice_selezionato.py
Risultato di quell'esperimento (solo 6 domande, esplorativo): fra due configurazioni
di ricerca la migliore in media peggiora circa un terzo dei casi; la media non dice
quali domande peggiorano.
"""

import sys
import time

from comuni import carica_domande, carica_fonti_giuste, leggi_chiave
from ricerca import Ricerca

MODELLO_GENERATIVO = "gemini-3.1-pro-preview"

# Tre istruzioni nel prompt contano più delle altre: usare SOLO i brani dati,
# citare il numero del brano per ogni affermazione, ammettere di non sapere.
MODELLO_PROMPT = """You are a financial assistant answering questions from an online forum. \
Use ONLY the numbered passages below to answer. Cite the passages you rely on by their \
number in square brackets, e.g. [3], for every factual claim. If the passages do not contain \
enough information to answer the question, say so explicitly instead of guessing.

Passages:
{brani}

Question: {domanda}

Answer:"""


def costruisci_prompt(domanda, testi_dei_brani):
    """Numera i brani [1], [2], ... e li inserisce nel modello di prompt."""
    righe = []
    for numero, testo in enumerate(testi_dei_brani, start=1):
        righe.append(f"[{numero}] {testo}")
    return MODELLO_PROMPT.format(brani="\n\n".join(righe), domanda=domanda)


def chiama_il_modello(prompt):
    """Manda il prompt a Gemini. Riprova fino a 3 volte, perché le reti falliscono."""
    from google import genai  # importato qui: serve solo con --chiama-api

    client = genai.Client(api_key=leggi_chiave())
    for tentativo in range(3):
        try:
            return client.models.generate_content(model=MODELLO_GENERATIVO, contents=prompt).text
        except Exception as errore:
            if tentativo == 2:
                return f"[ERRORE dopo 3 tentativi: {errore}]"
            time.sleep(5)


if __name__ == "__main__":
    ricerca = Ricerca()
    id_domanda = "dc1aaac0b33553d8c897d4150955d803<::>7"
    domanda = carica_domande("fiqa_rewrite")[id_domanda]
    ultima_frase = carica_domande("fiqa_lastturn")[id_domanda]
    fonti_giuste = carica_fonti_giuste()[id_domanda]

    posizioni_brani = ricerca.cerca([domanda, ultima_frase], quanti=10)
    prompt = costruisci_prompt(domanda, [ricerca.testi[i] for i in posizioni_brani])

    nel_contesto = sum(1 for i in posizioni_brani if ricerca.ids[i] in fonti_giuste)
    print(f"Fonti giuste presenti nei 10 brani passati al modello: {nel_contesto} su {len(fonti_giuste)}\n")
    print("----- PROMPT (abbreviato) -----")
    print(prompt[:1500] + "\n[...]\n")

    if "--chiama-api" in sys.argv:
        print("----- RISPOSTA DEL MODELLO -----")
        print(chiama_il_modello(prompt))
    else:
        print("(Nessuna chiamata al modello: aggiungi --chiama-api per farla.)")
