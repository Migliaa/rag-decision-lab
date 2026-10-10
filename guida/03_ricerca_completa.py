"""Esempio 3 — La ricerca completa: fusione di tre metodi e riordino, contro BGE da solo.

Cosa fa: prende una domanda delle 180, la cerca con BGE da solo e con la ricerca
completa, e mostra per ciascuna dove sono finite le fonti giuste (segnate con *).
Poi cerca anche una domanda scritta da te.

La logica è in guida/ricerca.py: leggi quel file prima di questo.

Come si lancia (dalla cartella "rag"; la preparazione richiede circa un minuto):
    uv run python guida/03_ricerca_completa.py
    uv run python guida/03_ricerca_completa.py "Why do I pay taxes on dividends?"

Corrisponde a: esperimenti/studio-embedding-retrieval/16_configurazione_finale.py

Che cosa dicono i numeri (da output/configurazione_finale.json, 180 domande):
  - BGE da solo: recall@10 0.417; solo la fusione dei metodi, senza riordino: 0.418;
    fusione + riordino sui primi 30: 0.478. Il guadagno viene quindi dal riordino,
    non dalla fusione.
  - Gli intervalli di confidenza (0.365-0.469 contro 0.426-0.532) si sovrappongono:
    il miglioramento è plausibile ma non è dimostrato con un test appaiato.

La domanda di esempio sotto è stata scelta fra le 12 del campione perché è l'unica
in cui la ricerca completa trova più fonti di BGE da solo (2 contro 1). Sulle stesse
12 domande il risultato è uguale in 10 casi e peggiore in 1: un esempio illustra il
meccanismo e non dice niente sulla qualità media.
"""

import sys

from comuni import carica_domande, carica_fonti_giuste
from ricerca import Ricerca

ricerca = Ricerca()


def mostra(titolo, posizioni_brani, fonti_giuste):
    print(f"=== {titolo} ===")
    for posto, i in enumerate(posizioni_brani, start=1):
        segno = "*" if ricerca.ids[i] in fonti_giuste else " "
        anteprima = " ".join(ricerca.testi[i].split())[:110]
        print(f"{posto:>2}. {segno} [{ricerca.ids[i]}] {anteprima}")
    trovate = sum(1 for i in posizioni_brani if ricerca.ids[i] in fonti_giuste)
    print(f"    fonti giuste nei primi {len(posizioni_brani)}: {trovate} su {len(fonti_giuste)}\n")


if len(sys.argv) > 1:
    # Domanda libera: non esistono fonti annotate, quindi niente asterischi.
    domanda = sys.argv[1]
    print(f"\nDOMANDA: {domanda}\n")
    mostra("BGE da solo", ricerca.solo_bge(domanda), set())
    mostra("Ricerca completa", ricerca.cerca([domanda]), set())
else:
    # Una domanda del dataset, con le sue due formulazioni.
    id_domanda = "dc1aaac0b33553d8c897d4150955d803<::>7"
    riscritta = carica_domande("fiqa_rewrite")[id_domanda]
    ultima_frase = carica_domande("fiqa_lastturn")[id_domanda]
    fonti_giuste = carica_fonti_giuste()[id_domanda]
    print(f"\nDOMANDA RISCRITTA: {riscritta}")
    print(f"ULTIMA FRASE:      {ultima_frase}\n")
    mostra("BGE da solo", ricerca.solo_bge(riscritta), fonti_giuste)
    mostra("Ricerca completa (due formulazioni)", ricerca.cerca([riscritta, ultima_frase]), fonti_giuste)
