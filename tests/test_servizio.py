"""Test del contratto HTTP con modello simulato: nessuna rete, nessun costo.

Verificano la forma di richieste, risposte ed errori. Non misurano la qualità
del recupero né della risposta.
"""

from unittest import SkipTest, TestCase

from fastapi.testclient import TestClient

from servizio.app import LUNGHEZZA_MASSIMA_DOMANDA, crea_app
from servizio.modello import EsitoModello, ModelloSimulato


class ModelloCheInventaUnaFonte:
    def genera(self, domanda, brani):
        return EsitoModello(risposta="testo", fonti=[brani[0].id, "mai-recuperato"])


class ContrattoTest(TestCase):
    def setUp(self) -> None:
        self.client = TestClient(crea_app())

    def test_health_risponde_ok(self) -> None:
        risposta = self.client.get("/health")
        self.assertEqual(risposta.status_code, 200)
        self.assertEqual(risposta.json(), {"status": "ok"})

    def test_domanda_valida_restituisce_risposta_fonti_e_request_id(self) -> None:
        risposta = self.client.post("/answer", json={"question": "What is an IRA?"})
        self.assertEqual(risposta.status_code, 200)
        corpo = risposta.json()
        self.assertEqual(set(corpo), {"answer", "sources", "request_id"})
        self.assertIn("What is an IRA?", corpo["answer"])
        self.assertEqual(corpo["sources"], ["simulato-1", "simulato-2"])
        self.assertTrue(corpo["request_id"])

    def test_due_richieste_hanno_request_id_diversi(self) -> None:
        primo = self.client.post("/answer", json={"question": "a"}).json()["request_id"]
        secondo = self.client.post("/answer", json={"question": "a"}).json()["request_id"]
        self.assertNotEqual(primo, secondo)

    def test_input_invalido_viene_rifiutato_con_422(self) -> None:
        casi = {
            "domanda vuota": {"question": ""},
            "soli spazi": {"question": "   "},
            "campo mancante": {},
            "tipo sbagliato": {"question": 42},
            "troppo lunga": {"question": "x" * (LUNGHEZZA_MASSIMA_DOMANDA + 1)},
        }
        for nome, corpo in casi.items():
            with self.subTest(nome):
                self.assertEqual(self.client.post("/answer", json=corpo).status_code, 422)

    def test_domanda_lunga_esattamente_il_limite_viene_accettata(self) -> None:
        risposta = self.client.post("/answer", json={"question": "x" * LUNGHEZZA_MASSIMA_DOMANDA})
        self.assertEqual(risposta.status_code, 200)

    def test_input_invalido_non_raggiunge_il_modello(self) -> None:
        # Con il modello guasto una domanda valida darebbe 503: ricevere 422
        # prova che la validazione ferma la richiesta prima.
        client = TestClient(crea_app(ModelloSimulato(guasto="indisponibile")))
        self.assertEqual(client.post("/answer", json={"question": ""}).status_code, 422)

    def test_guasti_del_modello_diventano_errori_senza_dettagli_interni(self) -> None:
        attesi = {"indisponibile": (503, "modello_non_disponibile"), "timeout": (504, "timeout_modello")}
        for guasto, (status, codice) in attesi.items():
            with self.subTest(guasto):
                client = TestClient(crea_app(ModelloSimulato(guasto=guasto)))
                risposta = client.post("/answer", json={"question": "What is an IRA?"})
                self.assertEqual(risposta.status_code, status)
                corpo = risposta.json()
                self.assertEqual(corpo["error"], codice)
                self.assertTrue(corpo["request_id"])
                self.assertNotIn("modello.interno", risposta.text)

    def test_fonte_non_recuperata_viene_scartata(self) -> None:
        client = TestClient(crea_app(ModelloCheInventaUnaFonte()))
        corpo = client.post("/answer", json={"question": "What is an IRA?"}).json()
        self.assertEqual(corpo["sources"], ["simulato-1"])

    def test_guasto_sconosciuto_viene_rifiutato_all_avvio(self) -> None:
        with self.assertRaises(ValueError):
            ModelloSimulato(guasto="altro")


class RecuperoRealeTest(TestCase):
    """Integrazione con modello e vettori locali; saltata se i dati non sono stati scaricati."""

    @classmethod
    def setUpClass(cls) -> None:
        from servizio import recupero

        if not (recupero.CORPUS.exists() and recupero.VETTORI.exists()):
            raise SkipTest("campione o vettori assenti: vedi scripts/scarica_dati.py")
        cls.recupero = recupero.RecuperoBge()
        cls.k = recupero.K

    def test_restituisce_k_brani_distinti_del_campione(self) -> None:
        brani = self.recupero("How is 401k account taxed?")
        self.assertEqual(len(brani), self.k)
        self.assertEqual(len({b.id for b in brani}), self.k)
        self.assertTrue({b.id for b in brani} <= {b.id for b in self.recupero.brani})

    def test_il_servizio_restituisce_le_fonti_del_recupero(self) -> None:
        client = TestClient(crea_app(recupero=self.recupero))
        corpo = client.post("/answer", json={"question": "How is 401k account taxed?"}).json()
        self.assertEqual(corpo["sources"], [b.id for b in self.recupero("How is 401k account taxed?")])


class DatabaseTest(TestCase):
    """Integrazione con PostgreSQL/pgvector locale; saltata se il database non è avviato."""

    @classmethod
    def setUpClass(cls) -> None:
        import psycopg

        from servizio import db, recupero

        if not (recupero.CORPUS.exists() and recupero.VETTORI.exists()):
            raise SkipTest("campione o vettori assenti: vedi scripts/scarica_dati.py")
        try:
            cls.connessione = db.connetti()
        except psycopg.OperationalError:
            raise SkipTest("database non raggiungibile: docker compose up -d db")
        cls.db = db
        cls.presenti = db.importa(cls.connessione)
        cls.in_memoria = recupero.RecuperoBge()
        cls.da_database = db.RecuperoPg()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connessione.close()

    def test_importazione_ripetuta_non_duplica(self) -> None:
        self.assertEqual(self.presenti, len(self.in_memoria.brani))
        self.assertEqual(self.db.importa(self.connessione), self.presenti)

    def test_ogni_riga_dichiara_la_provenienza(self) -> None:
        righe = self.connessione.execute("SELECT DISTINCT campione, modello, revisione FROM brani").fetchall()
        self.assertEqual(righe, [("E001-pilot-512", "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")])

    def test_il_database_restituisce_la_stessa_classifica_della_memoria(self) -> None:
        import json

        from servizio.recupero import CORPUS

        with (CORPUS.parent / "queries.jsonl").open(encoding="utf-8") as righe:
            domande = [json.loads(riga)["text"].removeprefix("|user|: ") for riga in righe if riga.strip()]
        for domanda in domande:
            with self.subTest(domanda):
                self.assertEqual(
                    [b.id for b in self.da_database(domanda)],
                    [b.id for b in self.in_memoria(domanda)],
                )
