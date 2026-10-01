"""Costruisce la vista interattiva V02/V03 con i dati reali della run."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/E001-20260912T141150Z-pilot512/figures/V02-V03-data.json"
DESTINATION = Path(r"C:\Users\andre\.codex\visualizations\2026\09\12\01a095cb-b272-70c2-89ab-ffe882eb773e\d1-embedding-explorer.html")

FRAGMENT = r'''
<div id="d1-embedding-explorer">
  <h1>La stessa query in MiniLM e BGE</h1>
  <div class="viz-controls">
    <label class="form-label" for="d1-query">Query MTRAG Human
      <select class="form-select" id="d1-query"></select>
    </label>
  </div>

  <div class="card" id="d1-query-detail" aria-live="polite"></div>

  <div class="d1-legend text-small">
    <span><b class="d1-dot d1-normal"></b> altro passaggio</span>
    <span><b class="d1-dot d1-top"></b> top 10 del modello</span>
    <span><b class="d1-ring"></b> fonte annotata pertinente</span>
    <span><b class="d1-star">★</b> query</span>
  </div>

  <div class="d1-plots">
    <section><h3>MiniLM</h3><div id="d1-minilm"></div></section>
    <section><h3>BGE-small</h3><div id="d1-bge"></div></section>
  </div>

  <div class="card text-small" id="d1-selected">Seleziona un punto per leggere il passaggio.</div>

  <h2>Prime cinque fonti recuperate</h2>
  <div class="d1-rankings" id="d1-rankings"></div>
  <p class="text-small text-muted">PCA calcolata separatamente per ogni encoder su 512 passaggi. Gli assi non hanno significato finanziario e le due mappe non condividono coordinate. Ranking e vicini sono calcolati nei 384 valori originali, non in 2D.</p>
  <div class="tooltip" role="tooltip" hidden></div>
</div>

<style>
#d1-embedding-explorer { width: 100%; color: var(--foreground); }
#d1-embedding-explorer .viz-controls { margin-block: 12px; }
#d1-embedding-explorer .form-label { min-width: min(100%, 560px); }
#d1-query-detail { padding: 12px; display: grid; gap: 8px; }
#d1-query-detail p { margin: 0; }
.d1-variant { display: grid; grid-template-columns: 90px 1fr; gap: 8px; }
.d1-legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0 4px; align-items: center; }
.d1-dot { width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 4px; }
.d1-normal { background: var(--muted-foreground); opacity: .45; }
.d1-top { background: var(--viz-series-1); }
.d1-ring { width: 12px; height: 12px; display: inline-block; border: 2px solid var(--foreground); border-radius: 50%; margin-right: 4px; }
.d1-star { color: var(--viz-series-3); }
.d1-plots { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.d1-plots section { min-width: 0; }
.d1-plots h3 { margin-bottom: 2px; }
.d1-chart { display: block; width: 100%; }
.d1-chart .axis path, .d1-chart .axis line, .d1-chart [data-chart-frame] { stroke: var(--border); }
.d1-chart .axis text, .d1-chart .axis-title { fill: var(--foreground); font-size: 12px; }
.d1-rankings { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }
.d1-rankings h3 { margin-bottom: 4px; }
.d1-rankings ol { padding-left: 24px; margin-top: 0; }
.d1-rankings li { margin-bottom: 9px; }
.d1-rankings code { overflow-wrap: anywhere; }
.d1-gold { color: var(--viz-series-3); font-weight: 500; }
#d1-embedding-explorer .tooltip { position: absolute; pointer-events: none; max-width: 340px; z-index: 2; padding: 8px; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); border-radius: 6px; }
@media (max-width: 720px) {
  .d1-plots, .d1-rankings { grid-template-columns: 1fr; }
  .d1-variant { grid-template-columns: 1fr; gap: 2px; }
}
</style>

<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const root = document.getElementById('d1-embedding-explorer');
  const DATA = __DATA__;
  const select = root.querySelector('#d1-query');
  const detail = root.querySelector('#d1-query-detail');
  const rankings = root.querySelector('#d1-rankings');
  const selected = root.querySelector('#d1-selected');
  const tooltip = root.querySelector('.tooltip');
  const charts = {};

  const clean = text => text.replaceAll('|user|:', '').trim();
  const short = (text, n=110) => text.length > n ? text.slice(0, n - 1) + '…' : text;
  DATA.queries.forEach(q => {
    const option = document.createElement('option');
    option.value = q.id;
    option.textContent = short(clean(q.lastturn), 85);
    select.appendChild(option);
  });
  const instructive = DATA.queries.find(q => q.id === 'dc1aaac0b33553d8c897d4150955d803<::>7');
  if (instructive) select.value = instructive.id;

  function currentQuery() { return DATA.queries.find(q => q.id === select.value); }

  function draw(method, containerId) {
    const container = root.querySelector(containerId);
    const width = Math.max(320, container.clientWidth || 440);
    const height = 350, margin = {top: 12, right: 18, bottom: 48, left: 64};
    container.replaceChildren();
    const svg = d3.select(container).append('svg')
      .attr('class', 'd1-chart').attr('viewBox', `0 0 ${width} ${height}`)
      .attr('role', 'img').attr('aria-label', `${method}: proiezione PCA dei passaggi e della query`);
    svg.append('title').text(`${method}: mappa PCA degli embedding reali`);
    svg.append('desc').text('La stella è la query, gli anelli sono le fonti annotate, i punti pieni sono i primi dieci risultati nello spazio originale.');
    const points = DATA.points[method];
    const q = currentQuery();
    const allX = points.map(d => d.x).concat(q.coordinates[method][0]);
    const allY = points.map(d => d.y).concat(q.coordinates[method][1]);
    const x = d3.scaleLinear().domain(d3.extent(allX)).nice().range([margin.left, width-margin.right]);
    const y = d3.scaleLinear().domain(d3.extent(allY)).nice().range([height-margin.bottom, margin.top]);
    svg.append('rect').attr('data-chart-frame','').attr('x',margin.left).attr('y',margin.top)
      .attr('width',width-margin.left-margin.right).attr('height',height-margin.top-margin.bottom).attr('fill','none');
    svg.append('g').attr('class','axis').attr('transform',`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(width < 400 ? 4 : 5));
    svg.append('g').attr('class','axis').attr('transform',`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5));
    svg.append('text').attr('class','axis-title').attr('data-axis','x').attr('x',(margin.left+width-margin.right)/2).attr('y',height-8).attr('text-anchor','middle').text('Componente PCA 1 (unità arbitrarie)');
    svg.append('text').attr('class','axis-title').attr('data-axis','y').attr('transform','rotate(-90)').attr('x',-(margin.top+height-margin.bottom)/2).attr('y',16).attr('text-anchor','middle').text('Componente PCA 2 (unità arbitrarie)');

    const topIds = new Set(q.rankings[method].map(d => d.id));
    const goldIds = new Set(q.relevant_ids);
    svg.append('g').selectAll('circle').data(points).join('circle')
      .attr('cx',d=>x(d.x)).attr('cy',d=>y(d.y))
      .attr('r',d=>goldIds.has(d.id)?4.5:(topIds.has(d.id)?3.5:2.2))
      .attr('fill',d=>topIds.has(d.id)?`var(--viz-series-${method==='minilm'?1:2})`:'var(--muted-foreground)')
      .attr('fill-opacity',d=>topIds.has(d.id)?0.9:0.28)
      .attr('stroke',d=>goldIds.has(d.id)?'var(--foreground)':'none').attr('stroke-width',2);
    const star = d3.symbol().type(d3.symbolStar).size(150)();
    svg.append('path').attr('d',star).attr('transform',`translate(${x(q.coordinates[method][0])},${y(q.coordinates[method][1])})`).attr('fill','var(--viz-series-3)');

    const screen = points.map(d => ({...d, sx:x(d.x), sy:y(d.y)}));
    const delaunay = d3.Delaunay.from(screen, d=>d.sx, d=>d.sy);
    svg.append('rect').attr('data-chart-hit','').attr('x',margin.left).attr('y',margin.top)
      .attr('width',width-margin.left-margin.right).attr('height',height-margin.top-margin.bottom)
      .attr('fill','transparent').style('pointer-events','all')
      .on('pointermove', event => {
        const [mx,my] = d3.pointer(event, svg.node()); const d = screen[delaunay.find(mx,my)];
        tooltip.hidden=false; tooltip.textContent=`${d.id} — ${short(d.text,220)}`;
        tooltip.style.left=`${event.clientX-root.getBoundingClientRect().left+12}px`;
        tooltip.style.top=`${event.clientY-root.getBoundingClientRect().top+12}px`;
      }).on('pointerleave',()=>tooltip.hidden=true)
      .on('click', event => {
        const [mx,my] = d3.pointer(event, svg.node()); const d=screen[delaunay.find(mx,my)];
        const ranks=['bm25','minilm','bge'].map(m=>{const r=q.rankings[m].find(x=>x.id===d.id);return `${m.toUpperCase()}: ${r?'#'+r.rank:'fuori top 5'}`}).join(' · ');
        selected.innerHTML=`<strong>${d.id}</strong><br>${d.text}<br><span class="text-muted">${ranks}</span>`;
      });
    charts[method] = {containerId};
  }

  function render() {
    const q=currentQuery();
    detail.innerHTML=`<p class="d1-variant"><strong>Ultimo turno</strong><span>${clean(q.lastturn)}</span></p><p class="d1-variant"><strong>Storia</strong><span>${clean(q.history).replaceAll('\n',' → ')}</span></p><p class="d1-variant"><strong>Riscrittura IBM</strong><span>${clean(q.rewrite)}</span></p>`;
    rankings.innerHTML=['bm25','minilm','bge'].map(method=>`<section><h3>${method==='bm25'?'BM25':method==='minilm'?'MiniLM':'BGE-small'}</h3><ol>${q.rankings[method].map(r=>`<li><code>${r.id}</code>${r.relevant?' <span class="d1-gold">✓ pertinente</span>':''}<br><span class="text-small">${r.text}</span></li>`).join('')}</ol></section>`).join('');
    selected.textContent='Seleziona un punto per leggere il passaggio.';
    draw('minilm','#d1-minilm'); draw('bge','#d1-bge');
  }
  select.addEventListener('change',render);
  new ResizeObserver(()=>{ clearTimeout(root._resizeTimer); root._resizeTimer=setTimeout(render,120); }).observe(root);
  render();
})();
</script>
'''


def main() -> None:
    data = SOURCE.read_text(encoding="utf-8")
    DESTINATION.write_text(FRAGMENT.replace("__DATA__", data), encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
