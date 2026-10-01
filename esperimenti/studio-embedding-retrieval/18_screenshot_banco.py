"""Step 18: catturare le figure del banco decisioni per il report.

Il banco (decisioni_m1.html) e' lo strumento con cui sono state prese le scelte di
progetto di M1, quindi nel report va mostrato lui, non solo i suoi risultati. Qui se ne
catturano quattro viste: la pagina intera, il modulo dove i criteri sono piu' lontani
dalle metriche di ranking (unita' di indicizzazione), il modulo con il catalogo completo
dei reranker, e la stessa pagina con i vincoli attivi che spengono le opzioni non
praticabili.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-embedding-retrieval/18_screenshot_banco.py
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PAGE = (REPO_ROOT / "strumenti" / "banco-decisioni-rag" / "banco_decisioni_rag.html").as_uri()
OUT = HERE / "output"
WIDTH = 1500


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": 1000}, device_scale_factor=2)
        page.goto(PAGE)
        page.wait_for_timeout(1200)  # attesa dei font remoti

        page.screenshot(path=str(OUT / "banco_intero.png"), full_page=True)
        print("banco_intero.png")

        for idx, name in ((0, "banco_unita_indicizzazione"), (5, "banco_reranker")):
            sec = page.locator("main section").nth(idx)
            sec.screenshot(path=str(OUT / f"{name}.png"))
            print(f"{name}.png")

        # con i vincoli attivi: le opzioni non praticabili si spengono e dichiarano il motivo
        for label in ("dati riservati", "budget zero"):
            page.get_by_role("button", name=label).click()
        page.wait_for_timeout(300)
        page.locator("main section").nth(5).screenshot(path=str(OUT / "banco_reranker_vincolato.png"))
        print("banco_reranker_vincolato.png")
        page.screenshot(path=str(OUT / "banco_intero_vincolato.png"), full_page=True)
        print("banco_intero_vincolato.png")

        browser.close()


if __name__ == "__main__":
    main()
