const state = {
  data: null,
  search: "",
  group: "",
  mode: "",
  sort: "name",
  window: 50,
  expanded: new Set()
};

const els = {
  heroMeta: document.querySelector("#hero-meta"),
  summaryGrid: document.querySelector("#summary-grid"),
  scoreChart: document.querySelector("#score-chart"),
  table: document.querySelector("#comparison-table"),
  search: document.querySelector("#search"),
  groupFilter: document.querySelector("#group-filter"),
  modeFilter: document.querySelector("#mode-filter"),
  runWindow: document.querySelector("#run-window"),
  sort: document.querySelector("#sort"),
  generatedAt: document.querySelector("#generated-at"),
  themeToggle: document.querySelector("#theme-toggle"),
  expandAll: document.querySelector("#expand-all"),
  collapseAll: document.querySelector("#collapse-all")
};

function applyTheme(theme) {
  const next = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (error) { /* ignore */ }
  if (els.themeToggle) {
    els.themeToggle.setAttribute("aria-pressed", String(next === "light"));
    els.themeToggle.innerHTML = next === "light"
      ? `<span aria-hidden="true">&#9789;</span> Dark`
      : `<span aria-hidden="true">&#9728;</span> Light`;
    els.themeToggle.title = next === "light" ? "Switch to dark mode" : "Switch to light mode";
  }
}

function initTheme() {
  let theme = null;
  try { theme = localStorage.getItem("theme"); } catch (error) { /* ignore */ }
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  applyTheme(theme);
}

function escAttr(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function formatNs(ns) {
  if (!Number.isFinite(ns)) return "-";
  if (ns < 1000) return `${ns.toFixed(1)} ns`;
  if (ns < 1_000_000) return `${(ns / 1000).toFixed(2)} us`;
  if (ns < 1_000_000_000) return `${(ns / 1_000_000).toFixed(2)} ms`;
  return `${(ns / 1_000_000_000).toFixed(2)} s`;
}

function formatRate(rate) {
  if (!Number.isFinite(rate) || rate <= 0) return "-";
  if (rate >= 1_000_000) return `${(rate / 1_000_000).toFixed(2)} Mops/s`;
  if (rate >= 1_000) return `${(rate / 1_000).toFixed(2)} Kops/s`;
  return `${rate.toFixed(2)} ops/s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "-";
  if (bytes === 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function benchmarkKey(result) {
  return `${result.group}||${result.name}||${result.unit || "op"}`;
}

function resultMap(run) {
  const map = new Map();
  for (const result of run.results || []) {
    map.set(benchmarkKey(result), result);
  }
  return map;
}

function selectedRuns() {
  const runs = state.data?.runs || [];
  return runs
    .filter((run) => !state.mode || run.mode === state.mode)
    .slice(0, state.window);
}

function filteredBenchmarks() {
  const search = state.search.toLowerCase();
  return (state.data?.benchmarks || []).filter((bench) => {
    if (state.group && bench.group !== state.group) return false;
    if (!search) return true;
    return `${bench.group} ${bench.name} ${bench.unit}`.toLowerCase().includes(search);
  });
}

function setOptions(select, values, selectedValue, allLabel) {
  select.innerHTML = "";
  const all = document.createElement("option");
  all.value = "";
  all.textContent = allLabel;
  select.appendChild(all);
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === selectedValue;
    select.appendChild(option);
  }
}

function renderSummary() {
  const runs = state.data.runs || [];
  const latest = runs[0];
  const commits = new Set(runs.map((run) => run.sha)).size;
  // Cases in the latest run, not the historical union of names across all runs
  // (renamed/removed benchmarks would otherwise inflate the count).
  const benchmarks = latest ? latest.case_count : (state.data.benchmarks?.length || 0);
  const best = runs.reduce((current, run) => run.score > (current?.score || 0) ? run : current, null);

  els.heroMeta.innerHTML = latest
    ? `<a href="${latest.run_url}" title="Open latest benchmark workflow">Latest run ${latest.short_sha}</a><span>${latest.mode}</span>`
    : "<span>No benchmark artifacts found yet</span>";

  const cards = [
    ["Runs", runs.length],
    ["Commits", commits],
    ["Benchmarks", benchmarks],
    ["Latest score", latest ? formatRate(latest.score) : "-"],
    ["Best score", best ? formatRate(best.score) : "-"],
    ["Latest commit", latest ? `<a href="${latest.commit_url}">${latest.short_sha}</a>` : "-"]
  ];

  els.summaryGrid.innerHTML = cards.map(([label, value]) => `
    <article class="summary-card">
      <span>${label}</span>
      <strong>${value}</strong>
    </article>
  `).join("");

  els.generatedAt.textContent = state.data.generated_at
    ? `Generated ${formatDate(state.data.generated_at)}`
    : "";
}

function renderChart() {
  const runs = selectedRuns().slice().reverse();
  // Fixed width regardless of commit count; points compress to fit.
  const width = 1180;
  const height = 260;
  const pad = { top: 24, right: 28, bottom: 56, left: 76 };
  const scores = runs.map((run) => run.score).filter((score) => Number.isFinite(score));

  if (runs.length === 0 || scores.length === 0) {
    els.scoreChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    els.scoreChart.innerHTML = `<text x="24" y="48" class="empty-text">No score data available.</text>`;
    return;
  }

  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const span = max - min || max || 1;
  const xStep = runs.length === 1 ? 0 : (width - pad.left - pad.right) / (runs.length - 1);
  const yScale = (score) => pad.top + (1 - ((score - min) / span)) * (height - pad.top - pad.bottom);
  const xScale = (index) => pad.left + index * xStep;
  const points = runs.map((run, index) => `${xScale(index)},${yScale(run.score)}`).join(" ");

  // Keep x labels legible: show at most ~14 evenly spaced commits plus the newest.
  const labelEvery = Math.max(1, Math.ceil(runs.length / 14));
  const showDots = runs.length <= 60;
  const labels = runs.map((run, index) => {
    if (index % labelEvery !== 0 && index !== runs.length - 1) return "";
    return `
    <g>
      <line x1="${xScale(index)}" y1="${height - pad.bottom}" x2="${xScale(index)}" y2="${height - pad.bottom + 6}" />
      <text x="${xScale(index)}" y="${height - 24}" text-anchor="middle">${run.short_sha}</text>
    </g>
  `;
  }).join("");

  const circles = runs.map((run, index) => {
    if (!showDots && index !== runs.length - 1) return "";
    return `
    <a href="${run.run_url}">
      <circle cx="${xScale(index)}" cy="${yScale(run.score)}" r="5">
        <title>${run.short_sha}: ${formatRate(run.score)}</title>
      </circle>
    </a>
  `;
  }).join("");

  els.scoreChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  els.scoreChart.innerHTML = `
    <line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" />
    <line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" />
    <text class="axis-label" x="16" y="${pad.top + 6}">${formatRate(max)}</text>
    <text class="axis-label" x="16" y="${height - pad.bottom}">${formatRate(min)}</text>
    <polyline class="trend-line" points="${points}" />
    ${circles}
    <g class="x-labels">${labels}</g>
  `;
}

function deltaClass(a, b) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return "";
  const pct = ((a - b) / b) * 100;
  if (pct <= -2) return "faster";
  if (pct >= 2) return "slower";
  return "same";
}

function deltaText(a, b) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return "-";
  const pct = ((a - b) / b) * 100;
  if (Math.abs(pct) < 0.1) return "0.0%";
  return `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

// Inline sparkline of median time across runs (chronological). Lower = faster = lower on the chart.
function sparkline(series) {
  const W = 168;
  const H = 34;
  const p = 4;
  const finite = series.filter((v) => Number.isFinite(v));
  if (finite.length === 0) {
    return `<svg class="spark" viewBox="0 0 ${W} ${H}" aria-hidden="true"></svg>`;
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || max || 1;
  const n = series.length;
  const x = (i) => (n === 1 ? W / 2 : p + (i * (W - 2 * p)) / (n - 1));
  const y = (v) => p + (1 - (v - min) / span) * (H - 2 * p);

  const points = series
    .map((v, i) => (Number.isFinite(v) ? `${x(i).toFixed(1)},${y(v).toFixed(1)}` : null))
    .filter(Boolean)
    .join(" ");

  let lastIdx = -1;
  for (let i = series.length - 1; i >= 0; i -= 1) {
    if (Number.isFinite(series[i])) { lastIdx = i; break; }
  }
  let prev = NaN;
  for (let i = lastIdx - 1; i >= 0; i -= 1) {
    if (Number.isFinite(series[i])) { prev = series[i]; break; }
  }
  const cls = deltaClass(series[lastIdx], prev);
  const dotFill = cls === "faster" ? "var(--accent-2)" : cls === "slower" ? "var(--danger)" : "var(--accent)";

  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
    <polyline class="spark-line" points="${points}" />
    <circle class="spark-dot" cx="${x(lastIdx).toFixed(1)}" cy="${y(series[lastIdx]).toFixed(1)}" r="3" style="fill:${dotFill}" />
  </svg>`;
}

// Per-commit measurement table shown when a benchmark row is expanded.
function buildDetail(bench, runsNewest, mapsNewest) {
  const rows = [];
  for (let i = 0; i < runsNewest.length; i += 1) {
    const run = runsNewest[i];
    const r = mapsNewest[i].get(bench.key);
    if (!r) continue;
    let prevMedian = NaN;
    for (let j = i + 1; j < runsNewest.length; j += 1) {
      const pr = mapsNewest[j].get(bench.key);
      if (pr) { prevMedian = Number(pr.median_ns); break; }
    }
    const med = Number(r.median_ns);
    const cls = deltaClass(med, prevMedian);
    const commit = run.commit_url
      ? `<a href="${run.commit_url}">${run.short_sha}</a>`
      : run.short_sha;
    rows.push(`
      <tr>
        <td class="dt-commit">${commit}</td>
        <td class="dt-date">${formatDate(run.created_at)}</td>
        <td>${run.mode || "-"}</td>
        <td><strong>${formatNs(med)}</strong></td>
        <td>${formatNs(Number(r.min_ns))}</td>
        <td>${formatNs(Number(r.max_ns))}</td>
        <td>${formatBytes(Number(r.allocated_bytes))}</td>
        <td class="${cls}">${deltaText(med, prevMedian)}</td>
      </tr>
    `);
  }

  if (rows.length === 0) {
    return `<p class="detail-empty">No measurements for this benchmark in the selected window.</p>`;
  }

  return `
    <div class="detail-wrap">
      <table class="detail-table">
        <thead>
          <tr>
            <th>Commit</th>
            <th>Date</th>
            <th>Mode</th>
            <th>Median</th>
            <th>Min</th>
            <th>Max</th>
            <th>Allocated</th>
            <th>&Delta; vs prev</th>
          </tr>
        </thead>
        <tbody>${rows.join("")}</tbody>
      </table>
    </div>
  `;
}

// Summary statistics for one benchmark across the (chronological) selected runs.
function benchStats(bench, mapsChrono) {
  const series = mapsChrono.map((m) => {
    const r = m.get(bench.key);
    return r ? Number(r.median_ns) : NaN;
  });
  const mins = mapsChrono.map((m) => {
    const r = m.get(bench.key);
    return r ? Number(r.min_ns) : NaN;
  });

  let latestIdx = -1;
  for (let i = series.length - 1; i >= 0; i -= 1) {
    if (Number.isFinite(series[i])) { latestIdx = i; break; }
  }
  let prev = NaN;
  for (let i = latestIdx - 1; i >= 0; i -= 1) {
    if (Number.isFinite(series[i])) { prev = series[i]; break; }
  }

  const latest = latestIdx === -1 ? NaN : series[latestIdx];
  const latestMin = latestIdx === -1 ? NaN : mins[latestIdx];
  const finite = series.filter((v) => Number.isFinite(v));
  const best = finite.length ? Math.min(...finite) : NaN;
  const deltaPct = (Number.isFinite(latest) && Number.isFinite(prev) && prev !== 0)
    ? ((latest - prev) / prev) * 100
    : NaN;
  const regressed = Number.isFinite(best) && best > 0 && latest >= best * 1.1;

  return { series, latest, latestMin, prev, best, deltaPct, latestIdx, regressed };
}

// NaN sorts last regardless of direction so missing/single-run benchmarks settle
// at the bottom instead of jumping to the top.
function compareWithNaNLast(a, b, dir) {
  const aNaN = !Number.isFinite(a);
  const bNaN = !Number.isFinite(b);
  if (aNaN && bNaN) return 0;
  if (aNaN) return 1;
  if (bNaN) return -1;
  return dir === "asc" ? a - b : b - a;
}

function sortBenches(entries) {
  const mode = state.sort;
  if (mode === "name") return entries; // already group+name ordered in the data
  const sorted = entries.slice();
  if (mode === "regression") {
    sorted.sort((x, y) => compareWithNaNLast(x.stats.deltaPct, y.stats.deltaPct, "desc"));
  } else if (mode === "improvement") {
    sorted.sort((x, y) => compareWithNaNLast(x.stats.deltaPct, y.stats.deltaPct, "asc"));
  } else if (mode === "slowest") {
    sorted.sort((x, y) => compareWithNaNLast(x.stats.latest, y.stats.latest, "desc"));
  } else if (mode === "fastest") {
    sorted.sort((x, y) => compareWithNaNLast(x.stats.latest, y.stats.latest, "asc"));
  }
  return sorted;
}

function renderTable() {
  const benches = filteredBenchmarks();
  const runsNewest = selectedRuns();
  const mapsNewest = runsNewest.map(resultMap);
  const mapsChrono = mapsNewest.slice().reverse(); // chronological, for the sparkline

  if (runsNewest.length === 0) {
    els.table.innerHTML = `<tbody><tr><td class="empty-cell">No benchmark runs match the current filters.</td></tr></tbody>`;
    return;
  }

  const header = `
    <thead>
      <tr>
        <th class="sticky-col">Benchmark</th>
        <th>Latest</th>
        <th>&Delta; vs prev</th>
        <th>Best</th>
        <th class="trend-col">Trend (${runsNewest.length} run${runsNewest.length === 1 ? "" : "s"})</th>
      </tr>
    </thead>
  `;

  const entries = sortBenches(benches.map((bench) => ({ bench, stats: benchStats(bench, mapsChrono) })));

  const body = entries.map(({ bench, stats }) => {
    const expanded = state.expanded.has(bench.key);
    const labelCell = `
      <th class="sticky-col">
        <span class="row-head">
          <span class="chevron" aria-hidden="true">&#9656;</span>
          <span class="bench-meta">
            <span>${bench.group}</span>
            <strong>${bench.name}</strong>
            <small>${bench.unit}</small>
          </span>
        </span>
      </th>`;

    const open = (cells) => `
      <tr class="bench-row${expanded ? " is-open" : ""}" data-key="${escAttr(bench.key)}" tabindex="0" role="button" aria-expanded="${expanded}">
        ${labelCell}${cells}
      </tr>
      ${expanded ? `<tr class="detail-row"><td class="detail-cell" colspan="5">${buildDetail(bench, runsNewest, mapsNewest)}</td></tr>` : ""}
    `;

    if (stats.latestIdx === -1) {
      return open(`<td class="missing">-</td><td class="missing">-</td><td class="missing">-</td><td class="missing">-</td>`);
    }

    const cls = deltaClass(stats.latest, stats.prev);
    const delta = deltaText(stats.latest, stats.prev);

    return open(`
      <td><strong>${formatNs(stats.latest)}</strong><span>min ${formatNs(stats.latestMin)}</span></td>
      <td class="${cls}">${delta}</td>
      <td><strong>${formatNs(stats.best)}</strong></td>
      <td class="trend-cell">${sparkline(stats.series)}${stats.regressed ? `<span class="warn-flag" title="Latest is &ge;10% slower than the best run">&#9888; slow</span>` : ""}</td>
    `);
  }).join("");

  els.table.innerHTML = `${header}<tbody>${body}</tbody>`;
}

function toggleRow(key) {
  if (!key) return;
  if (state.expanded.has(key)) {
    state.expanded.delete(key);
  } else {
    state.expanded.add(key);
  }
  renderTable();
}

function render() {
  renderSummary();
  renderChart();
  renderTable();
}

function wireControls() {
  els.search.addEventListener("input", (event) => {
    state.search = event.target.value;
    renderTable();
  });
  els.groupFilter.addEventListener("change", (event) => {
    state.group = event.target.value;
    renderTable();
  });
  els.modeFilter.addEventListener("change", (event) => {
    state.mode = event.target.value;
    renderSummary();
    renderChart();
    renderTable();
  });
  els.runWindow.addEventListener("change", (event) => {
    state.window = Number(event.target.value);
    renderChart();
    renderTable();
  });
  if (els.sort) {
    els.sort.addEventListener("change", (event) => {
      state.sort = event.target.value;
      renderTable();
    });
  }

  if (els.themeToggle) {
    els.themeToggle.addEventListener("click", () => {
      applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
    });
  }
  if (els.expandAll) {
    els.expandAll.addEventListener("click", () => {
      for (const bench of filteredBenchmarks()) state.expanded.add(bench.key);
      renderTable();
    });
  }
  if (els.collapseAll) {
    els.collapseAll.addEventListener("click", () => {
      state.expanded.clear();
      renderTable();
    });
  }

  // Row expand/collapse via click or keyboard (delegated; links stay clickable).
  els.table.addEventListener("click", (event) => {
    if (event.target.closest("a")) return;
    const row = event.target.closest("tr.bench-row");
    if (row && els.table.contains(row)) toggleRow(row.dataset.key);
  });
  els.table.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") return;
    const row = event.target.closest && event.target.closest("tr.bench-row");
    if (row && els.table.contains(row)) {
      event.preventDefault();
      toggleRow(row.dataset.key);
    }
  });
}

async function init() {
  initTheme();
  wireControls();
  try {
    const response = await fetch("data/benchmarks.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();

    const groups = [...new Set((state.data.benchmarks || []).map((bench) => bench.group))].sort();
    const modes = [...new Set((state.data.runs || []).map((run) => run.mode).filter(Boolean))].sort();
    setOptions(els.groupFilter, groups, state.group, "All groups");
    setOptions(els.modeFilter, modes, state.mode, "All modes");
    render();
  } catch (error) {
    els.heroMeta.innerHTML = `<span>Could not load benchmark data: ${error.message}</span>`;
    els.summaryGrid.innerHTML = "";
    els.table.innerHTML = `<tbody><tr><td class="empty-cell">Benchmark data is not available yet.</td></tr></tbody>`;
  }
}

init();
