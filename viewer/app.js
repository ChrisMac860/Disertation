/* Main viewer script; client-side only. */

const ui = {
  zipInput: document.getElementById('zipInput'),
  csvInput: document.getElementById('csvInput'),
  metaInput: document.getElementById('metaInput'),
  autoReloadBtn: document.getElementById('autoReloadBtn'),
  folderBtn: document.getElementById('folderBtn'),
  status: document.getElementById('status'),

  kind: {
    salinity: document.getElementById('kind-salinity'),
    water_level: document.getElementById('kind-water_level'),
    discharge: document.getElementById('kind-discharge'),
    rain: document.getElementById('kind-rain'),
  },

  stationSearch: document.getElementById('stationSearch'),
  stationSelect: document.getElementById('stationSelect'),
  dateStart: document.getElementById('dateStart'),
  dateEnd: document.getElementById('dateEnd'),
  thr3: document.getElementById('thr3'),
  thr4: document.getElementById('thr4'),
  markExceed: document.getElementById('markExceed'),
  resetBtn: document.getElementById('resetBtn'),
  exportBtn: document.getElementById('exportBtn'),
  exportMlBtn: document.getElementById('exportMlBtn'),

  overviewTableWrap: document.getElementById('overviewTableWrap'),
  charts: {
    salinity: document.getElementById('chart-salinity'),
    water_level: document.getElementById('chart-water_level'),
    discharge: document.getElementById('chart-discharge'),
    rain: document.getElementById('chart-rain'),
    missing: document.getElementById('chart-missingness'),
  },
  exceedTableWrap: document.getElementById('exceedTableWrap'),
  mockBtn: document.getElementById('mockBtn'),
};

const VALUE_COLUMN_HINTS = {
  salinity: ['sal', 'salinity', 'ppt', 'psu', 'g/l', 'g\\l', 'ec', 'ms/cm', 'value'],
  water_level: ['wl', 'water_level', 'level', 'stage', 'h', 'value'],
  discharge: ['q', 'discharge', 'flow', 'value'],
  rain: ['rain', 'rainfall', 'precip', 'ppt', 'mm', 'value'],
};

const DATETIME_COLUMN_HINTS = ['datetime', 'timestamp', 'date_time', 'time', 'date'];

const KIND_LABEL = {
  salinity: 'Salinity',
  water_level: 'Water level',
  discharge: 'Discharge',
  rain: 'Rain',
};

const HOUR_MS = 60 * 60 * 1000;
const ROOT_PREFIX = window.location.pathname.includes('/viewer/') ? '../' : '';

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function setStatus(msg) {
  ui.status.textContent = msg;
}

function formatNumber(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return '';
  return new Intl.NumberFormat('en-GB').format(n);
}

function pad(n) { return n < 10 ? '0' + n : '' + n; }
function toLocalISO(date) {
  const y = date.getFullYear();
  const m = pad(date.getMonth() + 1);
  const d = pad(date.getDate());
  const hh = pad(date.getHours());
  const mm = pad(date.getMinutes());
  const ss = pad(date.getSeconds());
  return `${y}-${m}-${d}T${hh}:${mm}:${ss}`;
}

function ymd(date) {
  const y = date.getFullYear();
  const m = pad(date.getMonth() + 1);
  const d = pad(date.getDate());
  return `${y}-${m}-${d}`;
}

function floorToHour(date) {
  const d = new Date(date.getTime());
  d.setMinutes(0, 0, 0);
  return d;
}

function parseDateFlexible(v) {
  if (v == null) return null;
  if (v instanceof Date && !isNaN(v)) return v;
  const s = String(v).trim();
  if (!s) return null;
  const d = new Date(s);
  if (!isNaN(d)) return d;
  if (s.includes(' ')) {
    const d2 = new Date(s.replace(' ', 'T'));
    if (!isNaN(d2)) return d2;
  }
  const m = s.match(/^(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})/);
  if (m) {
    const d3 = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
    if (!isNaN(d3)) return d3;
  }
  return null;
}

function detectDatetimeCol(columns, sampleRows) {
  const colsLower = columns.map(c => (c || '').toString().toLowerCase());
  for (let i = 0; i < colsLower.length; i++) {
    if (DATETIME_COLUMN_HINTS.some(k => colsLower[i].includes(k))) {
      const ok = sampleRows.filter(r => parseDateFlexible(r[columns[i]])).length;
      if (ok >= Math.min(50, Math.ceil(sampleRows.length * 0.6))) return columns[i];
    }
  }
  for (let i = 0; i < columns.length; i++) {
    const ok = sampleRows.filter(r => parseDateFlexible(r[columns[i]])).length;
    if (ok >= Math.min(50, Math.ceil(sampleRows.length * 0.6))) return columns[i];
  }
  return null;
}

function isNumeric(v) {
  if (v == null) return false;
  const n = typeof v === 'number' ? v : parseFloat(String(v).replace(/,/g, ''));
  return Number.isFinite(n);
}

function detectValueCol(columns, sampleRows, kind, dateCol) {
  const lowerCols = columns.map(c => (c || '').toString().toLowerCase());
  const prefer = VALUE_COLUMN_HINTS[kind] || [];
  for (let i = 0; i < columns.length; i++) {
    if (columns[i] === dateCol) continue;
    const c = lowerCols[i];
    if (prefer.some(k => c.includes(k))) {
      const num = sampleRows.filter(r => isNumeric(r[columns[i]])).length;
      if (num >= Math.min(10, Math.ceil(sampleRows.length * 0.2))) return columns[i];
    }
  }
  for (let i = 0; i < columns.length; i++) {
    const col = columns[i];
    if (col === dateCol) continue;
    const num = sampleRows.filter(r => isNumeric(r[col])).length;
    if (num >= Math.min(10, Math.ceil(sampleRows.length * 0.2))) return col;
  }
  return null;
}

function classifyKind(pathOrName) {
  const s = pathOrName.toLowerCase();
  if (s.includes('salinity')) return 'salinity';
  if (s.includes('/wl/') || s.includes('\\wl\\') || s.includes('water level')) return 'water_level';
  if (s.includes('/q/') || s.includes('\\q\\') || s.includes('discharge')) return 'discharge';
  if (s.includes('/pre/') || s.includes('\\pre\\') || s.includes('rain') || s.includes('rainfall')) return 'rain';
  return 'other';
}

function extractStationId(filename) {
  const name = filename.replace(/^[^/\\]*[\\/]/g, '').replace(/\.[^.]+$/, '');
  const bracket = name.match(/\[([^\]]+)\]/);
  if (bracket && bracket[1]) {
    return bracket[1].replace(/\s+/g, '');
  }
  const m = name.match(/^(.*)_\d{4}_\d{4}$/);
  if (m) return m[1];
  return name;
}

function downloadCSV(filename, rows) {
  const header = ['datetime', 'station_id', 'display_name', 'kind', 'value'];
  const lines = [header.join(',')];
  for (const r of rows) {
    const vals = [r.datetimeISO, r.station_id, (r.display_name || r.station_id), r.kind, (r.value ?? '')];
    lines.push(vals.map(v => {
      if (v == null) return '';
      const s = String(v);
      return s.includes(',') || s.includes('"') ? '"' + s.replace(/"/g, '""') + '"' : s;
    }).join(','));
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}


const state = {
  rows: [],
  stations: new Set(),
  stationMeta: new Map(),
  kindsAvailable: new Set(),
  minDate: null,
  maxDate: null,
  selectedKinds: new Set(['salinity', 'water_level', 'discharge', 'rain']),
  selectedStations: new Set(),
  filterStart: null,
  filterEnd: null,
  thr3: 3.0,
  thr4: 4.0,
  markExceed: false,
};

const STORAGE_KEY = 'msv_state_v1';
function saveState() {
  try {
    const obj = {
      selectedKinds: Array.from(state.selectedKinds),
      selectedStations: Array.from(state.selectedStations),
      filterStart: state.filterStart ? ymd(state.filterStart) : null,
      filterEnd: state.filterEnd ? ymd(state.filterEnd) : null,
      thr3: state.thr3, thr4: state.thr4, markExceed: state.markExceed,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(obj));
  } catch {}
}
function loadState() {
  try {
    const obj = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    if (!obj) return;
    state.selectedKinds = new Set(obj.selectedKinds || ['salinity', 'water_level', 'discharge', 'rain']);
    state.selectedStations = new Set(obj.selectedStations || []);
    state.thr3 = Number(obj.thr3 ?? 3.0);
    state.thr4 = Number(obj.thr4 ?? 4.0);
    state.markExceed = !!obj.markExceed;
    if (obj.filterStart) state.filterStart = new Date(obj.filterStart);
    if (obj.filterEnd) state.filterEnd = new Date(obj.filterEnd);
  } catch {}
}


async function parseCSVText(text, virtualPath) {
  return new Promise((resolve) => {
    Papa.parse(text, {
      header: true,
      dynamicTyping: false,
      skipEmptyLines: true,
      worker: false,
      complete: (results) => {
        resolve({ data: results.data, meta: results.meta, errors: results.errors, virtualPath });
      },
    });
  });
}

function mapFileToRows(parsed, stationMetaMap) {
  const path = parsed.virtualPath || 'file.csv';
  const kind = classifyKind(path);
  if (!['salinity', 'water_level', 'discharge', 'rain'].includes(kind)) return [];
  const filename = path.split(/[/\\]/).pop();
  const station_id = extractStationId(filename);

  const data = parsed.data;
  if (!data || !data.length) return [];
  const columns = parsed.meta?.fields || Object.keys(data[0] || {});
  const sample = data.slice(0, Math.min(300, data.length));
  const dateCol = detectDatetimeCol(columns, sample) || columns[0];
  const valueCol = detectValueCol(columns, sample, kind, dateCol) || columns.find(c => c !== dateCol) || null;
  if (!dateCol || !valueCol) return [];

  const meta = stationMetaMap.get(station_id);
  const display_name = meta?.display_name || station_id;

  const out = [];
  for (const row of data) {
    const dt = parseDateFlexible(row[dateCol]);
    if (!dt) continue;
    const vRaw = row[valueCol];
    const v = typeof vRaw === 'number' ? vRaw : parseFloat(String(vRaw).replace(/,/g, ''));
    if (!Number.isFinite(v)) continue;
    out.push({
      datetimeDate: dt,
      datetimeISO: toLocalISO(dt),
      station_id,
      kind,
      value: v,
      display_name,
    });
  }
  return out;
}

async function loadFromZip(file) {
  setStatus('Reading ZIP...');
  const zip = await JSZip.loadAsync(file);
  const csvEntries = Object.values(zip.files).filter(f => !f.dir && f.name.toLowerCase().endsWith('.csv'));
  let total = 0;
  const rows = [];
  for (const entry of csvEntries) {
    setStatus(`Parsing ${entry.name}`);
    const text = await entry.async('string');
    const parsed = await parseCSVText(text, entry.name);
    rows.push(...mapFileToRows(parsed, state.stationMeta));
  }
  total = rows.length;
  setStatus(`Loaded ${formatNumber(total)} rows from ${csvEntries.length} CSV files.`);
  return rows;
}

async function loadFromCSVFiles(files) {
  const csvs = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.csv'));
  const rows = [];
  for (const f of csvs) {
    setStatus(`Parsing ${f.name}`);
    const text = await f.text();
    const parsed = await parseCSVText(text, f.name);
    rows.push(...mapFileToRows(parsed, state.stationMeta));
  }
  setStatus(`Loaded ${formatNumber(rows.length)} rows from ${csvs.length} CSV files.`);
  return rows;
}

async function loadStationMetadata(file) {
  const text = await file.text();
  const parsed = await new Promise((resolve) => {
    Papa.parse(text, {
      header: true, dynamicTyping: false, skipEmptyLines: true,
      complete: (res) => resolve(res),
    });
  });
  const map = new Map();
  for (const r of parsed.data || []) {
    const sid = (r.station_id || '').toString().trim();
    if (!sid) continue;
    map.set(sid, {
      station_id: sid,
      display_name: (r.display_name || sid).toString().trim(),
      lat: r.lat != null ? Number(r.lat) : null,
      lon: r.lon != null ? Number(r.lon) : null,
      river: (r.river || '').toString(),
      notes: (r.notes || '').toString(),
    });
  }
  return map;
}

async function loadFromInventoryCsv(url = `${ROOT_PREFIX}reports/data_inventory.csv`) {
  try {
    setStatus('Looking for reports/data_inventory.csv...');
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error('inventory not found');
    const text = await res.text();
    const parsed = await new Promise((resolve) => {
      Papa.parse(text, { header: true, dynamicTyping: false, skipEmptyLines: true, complete: resolve });
    });
    const items = (parsed.data || []).filter(r => (r.RelativePath || '').toLowerCase().endsWith('.csv'));
    const rows = [];
    let i = 0;
    for (const it of items) {
      i++;
      const rel = String(it.RelativePath || '').replace(/\\/g, '/');
      const path = rel.startsWith('/') ? rel.slice(1) : rel;
      if (!path.toLowerCase().startsWith('data/')) { continue; }
      setStatus(`Fetching ${i}/${items.length}: ${path}`);
      try {
        const resp = await fetch(encodeURI(`${ROOT_PREFIX}${path}`), { cache: 'no-store' });
        if (!resp.ok) continue;
        const csvText = await resp.text();
        const parsedFile = await parseCSVText(csvText, path);
        rows.push(...mapFileToRows(parsedFile, state.stationMeta));
      } catch (e) {
        console.warn('Failed to fetch', path, e);
      }
    }
    setStatus(`Loaded ${formatNumber(rows.length)} rows from ${items.length} CSV files.`);
    return rows;
  } catch (e) {
    console.warn('Auto inventory load failed:', e);
    return null;
  }
}

async function loadFromFolderPicker() {
  if (!window.showDirectoryPicker) throw new Error('Folder access API not supported in this browser.');
  setStatus('Pick the data/ folder...');
  const dir = await window.showDirectoryPicker();
  const rows = [];
  async function walk(dirHandle, prefix = '') {
    for await (const [name, handle] of dirHandle.entries()) {
      const nextPath = prefix ? `${prefix}/${name}` : name;
      if (handle.kind === 'file' && name.toLowerCase().endsWith('.csv')) {
        setStatus(`Reading ${nextPath}`);
        const file = await handle.getFile();
        const text = await file.text();
        const parsed = await parseCSVText(text, nextPath);
        rows.push(...mapFileToRows(parsed, state.stationMeta));
      } else if (handle.kind === 'directory') {
        await walk(handle, nextPath);
      }
    }
  }
  await walk(dir);
  setStatus(`Loaded ${formatNumber(rows.length)} rows from picked folder.`);
  return rows;
}

async function attemptAutoLoad() {
  const fromInv = await loadFromInventoryCsv();
  if (fromInv && fromInv.length) {
    state.rows = fromInv;
    populateAfterLoad();
    return true;
  }
  setStatus('Auto-load failed. Use Pick folder or Upload.');
  return false;
}


function updateStationsUI() {
  const stationsArr = Array.from(state.stations).sort((a, b) => a.localeCompare(b));
  const getDisplay = (sid) => state.stationMeta.get(sid)?.display_name || sid;
  const search = (ui.stationSearch.value || '').toLowerCase();

  const prevSel = new Set(Array.from(ui.stationSelect.selectedOptions).map(o => o.value));

  ui.stationSelect.innerHTML = '';
  for (const sid of stationsArr) {
    const label = getDisplay(sid);
    if (search && !label.toLowerCase().includes(search) && !sid.toLowerCase().includes(search)) continue;
    const opt = document.createElement('option');
    opt.value = sid; opt.textContent = label;
    if (state.selectedStations.has(sid) || prevSel.has(sid)) opt.selected = true;
    ui.stationSelect.appendChild(opt);
  }
}

function populateAfterLoad() {
  state.stations = new Set(state.rows.map(r => r.station_id));
  state.kindsAvailable = new Set(state.rows.map(r => r.kind));
  if (state.rows.length) {
    let minT = Infinity, maxT = -Infinity;
    for (const r of state.rows) {
      const t = r.datetimeDate.getTime();
      if (t < minT) minT = t;
      if (t > maxT) maxT = t;
    }
    const min = new Date(minT);
    const max = new Date(maxT);
    state.minDate = min; state.maxDate = max;

    ui.dateStart.min = ymd(min); ui.dateStart.max = ymd(max);
    ui.dateEnd.min = ymd(min); ui.dateEnd.max = ymd(max);

    if (!state.filterStart) state.filterStart = min;
    if (!state.filterEnd) state.filterEnd = max;
    ui.dateStart.value = ymd(state.filterStart);
    ui.dateEnd.value = ymd(state.filterEnd);

    if (!state.selectedStations.size) {
      const first20 = Array.from(state.stations).sort().slice(0, 20);
      state.selectedStations = new Set(first20);
    }
  }
  ui.kind.salinity.checked = state.selectedKinds.has('salinity');
  ui.kind.water_level.checked = state.selectedKinds.has('water_level');
  ui.kind.discharge.checked = state.selectedKinds.has('discharge');
  ui.kind.rain.checked = state.selectedKinds.has('rain');
  ui.thr3.value = state.thr3;
  ui.thr4.value = state.thr4;
  ui.markExceed.checked = state.markExceed;

  updateStationsUI();
  refreshAll();
}

function getActiveFilters() {
  const kinds = new Set([
    ...(ui.kind.salinity.checked ? ['salinity'] : []),
    ...(ui.kind.water_level.checked ? ['water_level'] : []),
    ...(ui.kind.discharge.checked ? ['discharge'] : []),
    ...(ui.kind.rain.checked ? ['rain'] : []),
  ]);

  const selectedStations = new Set(Array.from(ui.stationSelect.selectedOptions).map(o => o.value));
  let start = ui.dateStart.value ? new Date(ui.dateStart.value + 'T00:00:00') : (state.filterStart || state.minDate);
  let end = ui.dateEnd.value ? new Date(ui.dateEnd.value + 'T23:59:59') : (state.filterEnd || state.maxDate);
  if (!(start instanceof Date) || isNaN(start)) start = state.minDate || new Date(0);
  if (!(end instanceof Date) || isNaN(end)) end = state.maxDate || new Date(8640000000000000);
  const thr3 = Number(ui.thr3.value || 3.0);
  const thr4 = Number(ui.thr4.value || 4.0);
  const markExceed = ui.markExceed.checked;

  state.selectedKinds = kinds;
  state.selectedStations = selectedStations.size ? selectedStations : state.selectedStations;
  state.filterStart = start; state.filterEnd = end;
  state.thr3 = thr3; state.thr4 = thr4; state.markExceed = markExceed;
  saveState();

  return { kinds, selectedStations: state.selectedStations, start, end, thr3, thr4, markExceed };
}

function filterRows() {
  const { kinds, selectedStations, start, end } = getActiveFilters();
  const ks = kinds;
  const st = selectedStations;
  const sTime = start ? start.getTime() : -Infinity;
  const eTime = end ? end.getTime() : Infinity;
  return state.rows.filter(r => ks.has(r.kind) && st.has(r.station_id) && r.datetimeDate.getTime() >= sTime && r.datetimeDate.getTime() <= eTime);
}

const refreshAll = debounce(() => {
  const filtered = filterRows();
  ui.exportBtn.disabled = filtered.length === 0;
  if (ui.exportMlBtn) ui.exportMlBtn.disabled = filtered.length === 0;
  renderOverview(filtered);
  renderCharts(filtered);
  renderDiagnostics(filtered);
  setStatus(`${formatNumber(filtered.length)} rows filtered · ${state.selectedStations.size} stations`);
}, 200);


function renderOverview(rows) {
  const kinds = ['salinity', 'water_level', 'discharge', 'rain'];
  const data = kinds.map(k => {
    const rr = rows.filter(r => r.kind === k);
    const cnt = rr.length;
    let cov = '-';
    if (cnt) {
      let minT = Infinity, maxT = -Infinity;
      for (const r of rr) {
        const t = r.datetimeDate.getTime();
        if (t < minT) minT = t;
        if (t > maxT) maxT = t;
      }
      const min = new Date(minT);
      const max = new Date(maxT);
      cov = `${ymd(min)} → ${ymd(max)}`;
    }
    return { kind: KIND_LABEL[k], count: cnt, coverage: cov };
  });

  const html = `
    <table>
      <thead><tr><th>Variable</th><th>Points</th><th>Coverage</th><th>Stations selected</th></tr></thead>
      <tbody>
        ${data.map(d => `<tr><td>${d.kind}</td><td>${formatNumber(d.count)}</td><td>${d.coverage}</td><td>${state.selectedStations.size}</td></tr>`).join('')}
      </tbody>
    </table>`;
  ui.overviewTableWrap.innerHTML = html;
}

function makeLineTraces(rows, kind) {
  const byStation = new Map();
  for (const r of rows) {
    if (r.kind !== kind) continue;
    if (!byStation.has(r.station_id)) byStation.set(r.station_id, []);
    byStation.get(r.station_id).push(r);
  }
  const traces = [];
  for (const [sid, arr] of byStation.entries()) {
    arr.sort((a, b) => a.datetimeDate - b.datetimeDate);
    traces.push({
      type: 'scatter', mode: 'lines',
      x: arr.map(r => r.datetimeDate),
      y: arr.map(r => r.value),
      name: state.stationMeta.get(sid)?.display_name || sid,
      hovertemplate: '%{fullData.name}<br>%{x|%Y-%m-%d %H:%M}<br>' + KIND_LABEL[kind] + ': %{y}<extra></extra>',
    });
  }
  return traces;
}

function layoutFor(kind) {
  const title = KIND_LABEL[kind] + ' over time';
  const shapes = [];
  const annotations = [];
  if (kind === 'salinity' && state.markExceed) {
    for (const [c, label] of [[state.thr3, '≥3 g/L'], [state.thr4, '≥4 g/L']]) {
      shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: c, y1: c, line: { color: '#e11d48', width: 1, dash: 'dot' } });
      annotations.push({ xref: 'paper', x: 1.0, xanchor: 'right', y: c, yanchor: 'bottom', text: label, showarrow: false, font: { color: '#e11d48' } });
    }
  }
  return {
    title, margin: { l: 50, r: 20, t: 30, b: 35 },
    xaxis: { title: 'Date' }, yaxis: { title: KIND_LABEL[kind] },
    legend: { orientation: 'h', y: -0.2 },
    uirevision: 'keep', shapes, annotations,
  };
}

function renderCharts(rows) {
  for (const k of ['salinity', 'water_level', 'discharge', 'rain']) {
    const el = ui.charts[k];
    if (!el) continue;
    const traces = makeLineTraces(rows, k);
    const layout = layoutFor(k);
    Plotly.react(el, traces, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['toImage'] });
  }
}

function renderDiagnostics(rows) {
  const dailyMap = new Map();
  const byKey = new Map();
  for (const r of rows) {
    const d = ymd(r.datetimeDate);
    const k = `${r.station_id}|${r.kind}|${d}`;
    dailyMap.set(k, (dailyMap.get(k) || 0) + 1);
  }
  for (const [k, c] of dailyMap.entries()) {
    const [sid, kind, day] = k.split('|');
    const sk = `${sid}|${kind}`;
    if (!byKey.has(sk)) byKey.set(sk, new Map());
    byKey.get(sk).set(day, c);
  }
  const traces = [];
  for (const [sk, dayMap] of byKey.entries()) {
    const [sid, kind] = sk.split('|');
    const arr = Array.from(dayMap.entries()).sort((a, b) => a[0].localeCompare(b[0]));
    traces.push({
      type: 'scatter', mode: 'lines',
      x: arr.map(e => e[0]), y: arr.map(e => e[1]),
      name: `${state.stationMeta.get(sid)?.display_name || sid} (${KIND_LABEL[kind] || kind})`,
      hovertemplate: '%{fullData.name}<br>%{x}: %{y} rows<extra></extra>',
    });
  }
  Plotly.react(ui.charts.missing, traces, {
    title: 'Daily row counts (missingness)', margin: { l: 50, r: 20, t: 30, b: 50 },
    xaxis: { title: 'Date', type: 'date' }, yaxis: { title: 'Rows per day' }, legend: { orientation: 'h', y: -0.2 },
    uirevision: 'keep',
  }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['toImage'] });

  const srows = rows.filter(r => r.kind === 'salinity');
  const byStation = new Map();
  for (const r of srows) {
    const t = byStation.get(r.station_id) || { n_obs: 0, ge3: 0, ge4: 0 };
    t.n_obs++;
    if (r.value >= state.thr3) t.ge3++;
    if (r.value >= state.thr4) t.ge4++;
    byStation.set(r.station_id, t);
  }
  const rowsOut = Array.from(byStation.entries()).map(([sid, o]) => ({
    station_id: sid,
    display_name: state.stationMeta.get(sid)?.display_name || sid,
    n_obs: o.n_obs,
    ge3: o.ge3,
    ge4: o.ge4,
    ge3_frac: o.n_obs ? (o.ge3 / o.n_obs) : 0,
    ge4_frac: o.n_obs ? (o.ge4 / o.n_obs) : 0,
  })).sort((a, b) => b.ge4 - a.ge4).slice(0, 20);

  const tableHtml = `
    <table>
      <thead>
        <tr><th>Station</th><th>Observations</th><th>≥3 g/L</th><th>≥4 g/L</th><th>≥3 frac</th><th>≥4 frac</th></tr>
      </thead>
      <tbody>
        ${rowsOut.map(r => `<tr>
          <td>${r.display_name}</td>
          <td>${formatNumber(r.n_obs)}</td>
          <td>${formatNumber(r.ge3)}</td>
          <td>${formatNumber(r.ge4)}</td>
          <td>${r.ge3_frac.toFixed(3)}</td>
          <td>${r.ge4_frac.toFixed(3)}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;
  ui.exceedTableWrap.innerHTML = tableHtml;
}


function onInputModeChanged() {
  const mode = ui.inputModeRadios.find(r => r.checked)?.value || 'zip';
  ui.zipInputWrap.classList.toggle('hidden', mode !== 'zip');
  ui.csvInputWrap.classList.toggle('hidden', mode !== 'csvs');
}

async function onZipSelected(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const rows = await loadFromZip(file);
    state.rows = rows;
    populateAfterLoad();
  } catch (err) {
    console.error(err);
    setStatus('Failed to read ZIP.');
  }
}

async function onCSVsSelected(e) {
  const files = e.target.files;
  if (!files || !files.length) return;
  try {
    const rows = await loadFromCSVFiles(files);
    state.rows = rows;
    populateAfterLoad();
  } catch (err) {
    console.error(err);
    setStatus('Failed to read CSV files.');
  }
}

async function onMetaSelected(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    state.stationMeta = await loadStationMetadata(file);
    for (const r of state.rows) {
      r.display_name = state.stationMeta.get(r.station_id)?.display_name || r.station_id;
    }
    updateStationsUI();
    refreshAll();
    setStatus('Station metadata loaded.');
  } catch (err) {
    console.error(err);
    setStatus('Failed to parse station metadata.');
  }
}

function onFiltersChanged() { refreshAll(); }
function onStationSearch() { updateStationsUI(); }

function onReset() {
  ui.kind.salinity.checked = true;
  ui.kind.water_level.checked = true;
  ui.kind.discharge.checked = true;
  ui.kind.rain.checked = true;
  ui.stationSearch.value = '';
  const stationsArr = Array.from(state.stations).sort();
  ui.stationSelect.innerHTML = '';
  stationsArr.forEach((sid, i) => {
    const opt = document.createElement('option');
    opt.value = sid; opt.textContent = state.stationMeta.get(sid)?.display_name || sid;
    if (i < 20) opt.selected = true;
    ui.stationSelect.appendChild(opt);
  });
  if (state.minDate && state.maxDate) {
    ui.dateStart.value = ymd(state.minDate);
    ui.dateEnd.value = ymd(state.maxDate);
  }
  ui.thr3.value = '3.0';
  ui.thr4.value = '4.0';
  ui.markExceed.checked = false;
  refreshAll();
}

function onExport() {
  const rows = filterRows().map(r => ({
    datetimeISO: r.datetimeISO,
    station_id: r.station_id,
    display_name: r.display_name,
    kind: r.kind,
    value: r.value,
  }));
  const filename = `mekong_filtered_${Date.now()}.csv`;
  downloadCSV(filename, rows);
}

async function exportForML() {
  try {
    const active = getActiveFilters();
    const stationIds = Array.from(active.selectedStations);
    if (!stationIds.length) {
      setStatus('Select at least one station before exporting for ML.');
      return;
    }
    if (typeof JSZip === 'undefined') {
      setStatus('JSZip not available for ML export.');
      return;
    }
    const stationSet = new Set(stationIds);
    const startBound = active.start ? floorToHour(active.start).getTime() : null;
    const endBound = active.end ? floorToHour(active.end).getTime() : null;

    const salRows = state.rows.filter((r) => {
      if (r.kind !== 'salinity') return false;
      if (!stationSet.has(r.station_id)) return false;
      const t = floorToHour(r.datetimeDate).getTime();
      if (startBound !== null && t < startBound) return false;
      if (endBound !== null && t > endBound) return false;
      return true;
    });

    if (!salRows.length) {
      setStatus('No salinity data available for the current filters.');
      return;
    }

    const valueByTime = new Map();
    let minTime = Infinity;
    let maxTime = -Infinity;
    for (const row of salRows) {
      const ts = floorToHour(row.datetimeDate).getTime();
      if (!valueByTime.has(ts)) valueByTime.set(ts, new Map());
      valueByTime.get(ts).set(row.station_id, row.value);
      if (ts < minTime) minTime = ts;
      if (ts > maxTime) maxTime = ts;
    }

    if (!Number.isFinite(minTime) || !Number.isFinite(maxTime) || minTime > maxTime) {
      setStatus('Unable to determine time range for ML export.');
      return;
    }

    const header = ['datetimeISO', ...stationIds];
    const valueLines = [header.join(',')];
    const maskLines = [header.join(',')];

    for (let ts = minTime; ts <= maxTime; ts += HOUR_MS) {
      const iso = toLocalISO(new Date(ts));
      const stationValues = valueByTime.get(ts);
      const valueRow = [iso];
      const maskRow = [iso];
      for (const sid of stationIds) {
        const val = stationValues ? stationValues.get(sid) : undefined;
        const isPresent = Number.isFinite(val);
        valueRow.push(isPresent ? val : 'NaN');
        maskRow.push(isPresent ? '1' : '0');
      }
      valueLines.push(valueRow.join(','));
      maskLines.push(maskRow.join(','));
    }

    setStatus('Building training_data.zip ...');
    const zip = new JSZip();
    zip.file('tensor_values.csv', valueLines.join('\n'));
    zip.file('tensor_mask.csv', maskLines.join('\n'));
    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'training_data.zip';
    a.click();
    URL.revokeObjectURL(url);
    setStatus('training_data.zip downloaded.');
  } catch (err) {
    console.error('exportForML failed', err);
    setStatus('Failed to export ML tensors.');
  }
}

function generateMockData() {
  const stations = ['Alpha', 'Bravo', 'Charlie'];
  const kinds = ['salinity', 'water_level'];
  const start = new Date(); start.setHours(0,0,0,0); start.setDate(start.getDate() - 6);
  const rows = [];
  for (const sid of stations) {
    for (const kind of kinds) {
      for (let d = 0; d < 7; d++) {
        const day = new Date(start.getTime() + d*24*3600*1000);
        for (let h = 0; h < 24; h += 6) {
          const dt = new Date(day); dt.setHours(h);
          rows.push({
            datetimeDate: new Date(dt),
            datetimeISO: toLocalISO(dt),
            station_id: sid,
            kind,
            value: kind === 'salinity' ? Math.max(0, 2 + Math.sin(d/2 + h/10) * 3) : 1 + Math.random()*2,
            display_name: sid,
          });
        }
      }
    }
  }
  return rows;
}

function onMock() {
  state.rows = generateMockData();
  state.stationMeta = new Map();
  populateAfterLoad();
  setStatus('Mock data loaded.');
}


function init() {
  loadState();

  if (ui.autoReloadBtn) ui.autoReloadBtn.addEventListener('click', attemptAutoLoad);
  if (ui.folderBtn) ui.folderBtn.addEventListener('click', async () => {
    try {
      const rows = await loadFromFolderPicker();
      state.rows = rows;
      populateAfterLoad();
    } catch (e) {
      console.warn(e);
      setStatus('Folder access not available or cancelled.');
    }
  });

  if (ui.zipInput) ui.zipInput.addEventListener('change', onZipSelected);
  if (ui.csvInput) ui.csvInput.addEventListener('change', onCSVsSelected);
  if (ui.metaInput) ui.metaInput.addEventListener('change', onMetaSelected);

  ui.stationSearch.addEventListener('input', onStationSearch);
  ui.stationSelect.addEventListener('change', onFiltersChanged);
  ui.dateStart.addEventListener('change', onFiltersChanged);
  ui.dateEnd.addEventListener('change', onFiltersChanged);
  ui.kind.salinity.addEventListener('change', onFiltersChanged);
  ui.kind.water_level.addEventListener('change', onFiltersChanged);
  ui.kind.discharge.addEventListener('change', onFiltersChanged);
  ui.kind.rain.addEventListener('change', onFiltersChanged);
  ui.thr3.addEventListener('input', onFiltersChanged);
  ui.thr4.addEventListener('input', onFiltersChanged);
  ui.markExceed.addEventListener('change', onFiltersChanged);
  ui.resetBtn.addEventListener('click', onReset);
  ui.exportBtn.addEventListener('click', onExport);
  if (ui.exportMlBtn) ui.exportMlBtn.addEventListener('click', exportForML);
  if (ui.mockBtn) ui.mockBtn.addEventListener('click', onMock);

  ['salinity','water_level','discharge','rain'].forEach(k => Plotly.newPlot(ui.charts[k], [], layoutFor(k), {responsive:true, displaylogo:false}));
  Plotly.newPlot(ui.charts.missing, [], {title:'Daily row counts (missingness)'}, {responsive:true, displaylogo:false});

  attemptAutoLoad();
}

init();


