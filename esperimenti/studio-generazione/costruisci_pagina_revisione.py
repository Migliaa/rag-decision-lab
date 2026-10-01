"""Costruisce la pagina di revisione del pilot: un HTML locale, senza server, dove Andrea
compila a mano la griglia a quattro voci per ciascuna delle 5 risposte generate.

Non e' un artefatto pubblicato: e' uno strumento locale, sullo stesso modello del banco
decisioni (nessuna dipendenza, si apre col doppio clic). Lo stato compilato si salva in
localStorage del browser e si esporta in JSON con un pulsante, per essere riletto da
qualunque script successivo di M2.

Va rieseguito ogni volta che 03_pilot_generazione.py produce un nuovo pilot_generazione.json.

Eseguirlo con: .venv/Scripts/python.exe esperimenti/studio-generazione/costruisci_pagina_revisione.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "output"

HTML = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Revisione pilot — M2</title>
<style>
:root{{--bg:#faf9f7;--fg:#1f1c19;--muted:#6b6459;--line:#ddd6cc;--accent:#7a4b2e;--accent-bg:#f1e6dc;--ok:#3d7a4f;--bad:#a13a3a}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1815;--fg:#ece7e0;--muted:#a39a8c;--line:#3a352e;--accent:#d8b28c;--accent-bg:#2c241d;--ok:#7fbf8f;--bad:#e08080}}}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);font-family:"Iowan Old Style",Georgia,serif;margin:0;padding:24px 16px 80px;line-height:1.5}}
h1{{font-size:1.3rem;margin:0 0 4px}}
.sub{{color:var(--muted);font-size:.85rem;margin-bottom:24px}}
.card{{border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:20px;background:color-mix(in srgb, var(--bg) 96%, var(--fg))}}
.regime{{display:inline-block;font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;background:var(--accent-bg);color:var(--accent);padding:2px 8px;border-radius:5px;margin-bottom:8px}}
.domanda{{font-size:1.05rem;margin:4px 0 2px}}
.turno{{color:var(--muted);font-size:.82rem;margin-bottom:10px}}
details{{margin:10px 0}}
summary{{cursor:pointer;color:var(--accent);font-size:.85rem}}
.passaggio{{font-size:.82rem;padding:6px 0;border-bottom:1px solid var(--line)}}
.passaggio b{{color:var(--accent)}}
.risposta{{background:var(--accent-bg);border-radius:8px;padding:12px 14px;margin:10px 0;font-size:.95rem;white-space:pre-wrap}}
.grid{{display:grid;gap:10px;margin-top:12px}}
.q{{display:flex;align-items:baseline;gap:10px;font-size:.88rem}}
.q label{{flex:0 0 260px}}
.q .opts{{display:flex;gap:6px}}
.opts button{{border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;padding:4px 10px;font-size:.82rem;cursor:pointer;font-family:inherit}}
.opts button.sel-yes{{background:var(--ok);color:#fff;border-color:var(--ok)}}
.opts button.sel-no{{background:var(--bad);color:#fff;border-color:var(--bad)}}
textarea{{width:100%;min-height:50px;background:transparent;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-family:inherit;font-size:.85rem;margin-top:6px}}
.bar{{position:fixed;bottom:0;left:0;right:0;background:var(--bg);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;align-items:center}}
.bar button{{border:1px solid var(--accent);background:var(--accent);color:#fff;border-radius:6px;padding:8px 16px;font-size:.85rem;cursor:pointer;font-family:inherit}}
.bar span{{font-size:.8rem;color:var(--muted)}}
</style>
</head>
<body>
<h1>Revisione del pilot di generazione</h1>
<div class="sub">Griglia a quattro voci per ciascuna risposta. Salva in locale, esporta in JSON per Appunti2.</div>
<div id="cards"></div>
<div class="bar">
  <button onclick="salva()">Salva in questo browser</button>
  <button onclick="esporta()">Esporta JSON</button>
  <span id="stato"></span>
</div>
<script>
const DATA = {dati};
const DOMANDE = [
  ["basta","Il contesto contiene una risposta?"],
  ["fondata","La risposta e' fondata sul contesto (non inventa)?"],
  ["corretta","La risposta e' corretta?"],
  ["astiene","Se il contesto non bastava, lo ha dichiarato?"]
];
const KEY = "m2-revisione-pilot-v1";
let stato = {{}};
try {{ const r = localStorage.getItem(KEY); if(r) stato = JSON.parse(r); }} catch(e) {{}}

function render(){{
  const root = document.getElementById("cards");
  root.innerHTML = "";
  DATA.forEach((r,i) => {{
    stato[r.query_id] = stato[r.query_id] || {{note:""}};
    const s = stato[r.query_id];
    const passaggi = r.passaggi_nel_contesto.map((pid,n) => {{
      const annotata = r.fonti_annotate[pid];
      return `<div class="passaggio"><b>[${{n+1}}]</b> ${{pid}}${{annotata?` — annotata, posizione ${{annotata}}`:""}}</div>`;
    }}).join("");
    const domande = DOMANDE.map(([k,l]) => {{
      if(k==="astiene" && s.basta===true){{
        return `<div class="q"><label>${{l}}</label><div class="opts"><span style="color:var(--muted);font-size:.82rem">non applicabile: il contesto bastava</span></div></div>`;
      }}
      return `<div class="q"><label>${{l}}</label><div class="opts" data-k="${{k}}" data-q="${{r.query_id}}">
        <button class="${{s[k]===true?'sel-yes':''}}" onclick="set('${{r.query_id}}','${{k}}',true)">sì</button>
        <button class="${{s[k]===false?'sel-no':''}}" onclick="set('${{r.query_id}}','${{k}}',false)">no</button>
      </div></div>`;
    }}).join("");
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = `
      <span class="regime">Regime ${{r.regime}} — ${{r.etichetta}}</span>
      <div class="domanda">${{r.domanda_riscritta}}</div>
      <div class="turno">ultimo turno reale: «${{r.domanda_ultimo_turno}}»</div>
      <details><summary>10 passaggi nel contesto</summary>${{passaggi}}</details>
      <div class="risposta">${{r.risposta}}</div>
      <div class="grid">${{domande}}</div>
      <textarea placeholder="nota libera" onchange="setNote('${{r.query_id}}',this.value)">${{s.note||""}}</textarea>
    `;
    root.appendChild(div);
  }});
}}
function set(qid,k,v){{ stato[qid][k] = v; render(); }}
function setNote(qid,v){{ stato[qid].note = v; }}
function salva(){{
  try {{ localStorage.setItem(KEY, JSON.stringify(stato));
    document.getElementById("stato").textContent = "salvato " + new Date().toLocaleTimeString();
  }} catch(e) {{ document.getElementById("stato").textContent = "errore salvataggio locale"; }}
}}
function esporta(){{
  const blob = new Blob([JSON.stringify(stato,null,2)], {{type:"application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "revisione_pilot.json";
  a.click();
}}
render();
</script>
</body>
</html>
"""


def main() -> None:
    dati = json.loads((OUT_DIR / "pilot_generazione.json").read_text(encoding="utf-8"))
    out = OUT_DIR / "revisione_pilot.html"
    out.write_text(HTML.format(dati=json.dumps(dati, ensure_ascii=False)), encoding="utf-8")
    print(f"scritto: {out}")


if __name__ == "__main__":
    main()
