@echo off
rem Apre il banco decisioni RAG nel browser predefinito.
rem Funziona senza server: la pagina e' un singolo file HTML autonomo.
rem I dati modificati restano nel localStorage del browser usato per aprirlo,
rem quindi aprirlo sempre con lo stesso browser, oppure esportare in JSON
rem (pulsante "esporta json") e salvare il file in .\dati\ .

start "" "%~dp0banco_decisioni_rag.html"
