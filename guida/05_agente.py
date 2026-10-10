"""Esempio 5 — Un agente: un modello che decide da solo cosa cercare, passo dopo passo.

Nelle tappe precedenti la ricerca era fissa: una domanda, una ricerca, i brani. Qui il
modello riceve la domanda SENZA brani e due strumenti:
  - search_text("parole"): cerca i brani che contengono quelle parole;
  - read("id"): legge un brano per intero.
A ogni giro il modello scrive un pensiero e un'azione; il codice esegue l'azione e
gli restituisce l'esito (Observation). Il ciclo si ripete finché il modello scrive
"Final Answer" oppure finisce il numero massimo di passi.

Per vedere il meccanismo senza spendere niente, di default al posto del modello
gira un MODELLO FINTO scritto qui sotto, che segue sempre lo stesso copione (cerca,
legge il primo risultato, risponde). Gli strumenti invece sono veri e lavorano sul
corpus vero. Con --chiama-api al posto del finto c'è Gemini (a pagamento o con
quota, come nell'esempio 4).

Si vede anche il limite di questo strumento: la ricerca per sole parole restituisce
brani in cui "income" e "tax" compaiono, non quelli che rispondono alla domanda.

Come si lancia (dalla cartella "rag"; la preparazione richiede circa un minuto):
    uv run python guida/05_agente.py
    uv run --extra generazione python guida/05_agente.py --chiama-api

Corrisponde a: esperimenti/studio-agenti/01_agente_ricerca_testuale.py
Cosa è emerso lì (2 domande, pilota): vicino al limite di passi l'agente ha dato una
risposta sicura ma infondata, e su un'altra domanda è rimasto bloccato oltre venti
minuti perché il tetto era sui passi e non sul tempo.
"""

import re
import sys

from comuni import carica_corpus, leggi_chiave

# Il limite lo impone il CODICE, non il modello: di un agente non ci si fida che
# si fermi da solo. Un tetto sui passi non basta se ogni passo può durare minuti.
MASSIMO_PASSI = 6
CARATTERI_LETTI = 1500

DOMANDA = "Why do I still have to pay income tax?"

ISTRUZIONI = """You are a research agent answering a question using a search tool over a forum \
corpus. You do NOT get any passages upfront: you must search for them yourself.

At each turn, respond with EXACTLY one of these forms, nothing else:

Thought: <your reasoning>
Action: search_text["<query>"]

or

Thought: <your reasoning>
Action: read["<passage_id>"]

or, when you have enough information:

Thought: <your reasoning>
Final Answer: <your answer, citing passage ids you used, or stating the search did not find enough>

You have at most {massimo} actions before you must give a Final Answer."""


def prepara_strumenti():
    """Costruisce i due strumenti sul corpus vero."""
    ids, testi = carica_corpus()
    posto_del_brano = {id_brano: i for i, id_brano in enumerate(ids)}
    parole_dei_brani = [set(re.findall(r"[a-z0-9]+", t.lower())) for t in testi]

    def search_text(ricerca):
        """Conta quante parole della ricerca compaiono in ogni brano (nessun significato)."""
        parole_cercate = set(re.findall(r"[a-z0-9]+", ricerca.lower()))
        punteggi = [(len(parole_cercate & parole_dei_brani[i]), i) for i in range(len(ids))]
        punteggi.sort(key=lambda coppia: -coppia[0])
        righe = []
        for punteggio, i in punteggi[:5]:
            if punteggio > 0:
                anteprima = testi[i][:200].replace("\n", " ")
                righe.append(f"[{ids[i]}] {anteprima}...")
        return "\n".join(righe) if righe else "No results."

    def read(id_brano):
        if id_brano not in posto_del_brano:
            return f"Unknown passage id: {id_brano}"
        return testi[posto_del_brano[id_brano]][:CARATTERI_LETTI]

    return search_text, read


def modello_finto(trascrizione):
    """Segue un copione: giro 1 cerca, giro 2 legge il primo risultato, giro 3 risponde."""
    giri_fatti = sum(1 for riga in trascrizione if riga.startswith("Thought:"))
    if giri_fatti == 0:
        return 'Thought: Searching for the topic.\nAction: search_text["income tax"]'
    if giri_fatti == 1:
        primo_id = re.search(r"\[([^\]]+)\]", trascrizione[-1]).group(1)
        return f'Thought: Reading the first result.\nAction: read["{primo_id}"]'
    return "Thought: I have read one passage.\nFinal Answer: (risposta del modello finto, nessun contenuto reale)"


def modello_vero(trascrizione):
    from google import genai  # importato qui: serve solo con --chiama-api

    prompt = ISTRUZIONI.format(massimo=MASSIMO_PASSI) + "\n\n" + "\n\n".join(trascrizione)
    client = genai.Client(api_key=leggi_chiave())
    return client.models.generate_content(model="gemini-3.1-pro-preview", contents=prompt).text.strip()


AZIONE = re.compile(r'Action:\s*(search_text|read)\["(.*?)"\]', re.DOTALL)
RISPOSTA_FINALE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)

if __name__ == "__main__":
    print("Preparo gli strumenti sul corpus...")
    search_text, read = prepara_strumenti()
    chiedi_al_modello = modello_vero if "--chiama-api" in sys.argv else modello_finto

    # La trascrizione è tutto ciò che il modello "ricorda": a ogni giro gli viene
    # rimandata per intero, con in coda la nuova Observation.
    trascrizione = [f"Question: {DOMANDA}"]
    risposta_finale = None

    for passo in range(1, MASSIMO_PASSI + 1):
        output = chiedi_al_modello(trascrizione)
        trascrizione.append(output)
        print(f"--- passo {passo} ---\n{output}")

        finale = RISPOSTA_FINALE.search(output)
        if finale:
            risposta_finale = finale.group(1).strip()
            break

        azione = AZIONE.search(output)
        if azione:
            strumento, argomento = azione.group(1), azione.group(2)
            esito = search_text(argomento) if strumento == "search_text" else read(argomento)
        else:
            esito = "(unparseable response, please follow the exact format)"
        trascrizione.append(f"Observation: {esito}")
        print(f"Observation: {esito[:400]}\n")
    else:
        print(f"[fermato dal codice dopo {MASSIMO_PASSI} passi, nessuna Final Answer]")

    print("\nRISPOSTA FINALE:", risposta_finale or "nessuna")
