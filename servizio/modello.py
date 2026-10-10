"""Sostituti simulati di recupero e modello per la tappa A.

Nessuna rete, nessuna chiave, nessun costo. Servono a provare il contratto
HTTP: che cosa entra, che cosa esce e che cosa succede quando il modello
fallisce. NON dicono nulla sulla qualità di un recupero o di una risposta.

Il servizio conosce soltanto due cose di un modello: il metodo ``genera`` e le
due eccezioni qui sotto. Un modello vero, più avanti, dovrà rispettare la
stessa forma e ``app.py`` non cambierà.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Brano:
    id: str
    testo: str


@dataclass(frozen=True)
class EsitoModello:
    risposta: str
    fonti: list[str]


class ModelloNonDisponibile(Exception):
    """Il modello non è raggiungibile o rifiuta la richiesta."""


class ModelloTimeout(Exception):
    """Il modello non ha risposto entro il tempo concesso."""


# Brani inventati, con ID che dichiarano di esserlo: non vengono da FiQA.
BRANI_SIMULATI = [
    Brano("simulato-1", "An IRA is an individual retirement account with tax advantages."),
    Brano("simulato-2", "Contributions to a traditional IRA may be tax-deductible."),
]


def recupero_simulato(domanda: str) -> list[Brano]:
    # Restituisce sempre gli stessi brani, qualunque sia la domanda. Nella
    # tappa successiva questo è il punto in cui entra il recupero vero.
    return BRANI_SIMULATI


GUASTI = ("indisponibile", "timeout")


class ModelloSimulato:
    def __init__(self, guasto: str | None = None) -> None:
        if guasto is not None and guasto not in GUASTI:
            raise ValueError(f"guasto sconosciuto: {guasto!r}; ammessi: {GUASTI}")
        self.guasto = guasto

    def genera(self, domanda: str, brani: list[Brano]) -> EsitoModello:
        # I messaggi delle eccezioni contengono di proposito un dettaglio
        # interno (un indirizzo): deve finire nel log, mai nella risposta HTTP.
        if self.guasto == "indisponibile":
            raise ModelloNonDisponibile("connessione rifiutata da modello.interno:8080")
        if self.guasto == "timeout":
            # Il ritardo non è atteso davvero: si simula soltanto il suo esito.
            raise ModelloTimeout("nessuna risposta da modello.interno:8080 dopo 30 s")
        # Nessun testo viene generato: si riporta l'inizio del primo brano,
        # così si vede che cosa il recupero ha passato al modello.
        primo = " ".join(brani[0].testo.split())[:200] if brani else "nessun brano"
        return EsitoModello(
            risposta=f"[risposta simulata] Domanda: {domanda} | Primo brano recuperato: {primo}",
            fonti=[b.id for b in brani],
        )
