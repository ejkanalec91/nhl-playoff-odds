// Shared helpers: data loading, formatting, page chrome, and the line chart.

const SERIES_VARS = ["--series-1", "--series-2", "--series-3", "--series-4"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

async function loadJSON(name) {
  const res = await fetch(`data/${name}?t=${Date.now() / 3.6e6 | 0}`);
  if (!res.ok) throw new Error(`Could not load ${name}`);
  return res.json();
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmtPct(p) {
  if (p == null) return "–";
  if (p <= 0) return "0%";
  if (p >= 1) return "100%";
  if (p < 0.005) return "<1%";
  if (p > 0.995) return ">99%";
  return Math.round(p * 100) + "%";
}

function fmtSigned(x, digits = 2) {
  return (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(digits);
}

function fmtDate(iso, withWeekday = false) {
  const [y, m, d] = iso.split("-").map(Number);
  const s = `${MONTHS[m - 1]} ${d}`;
  if (!withWeekday) return s;
  const wd = new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" });
  return `${wd}, ${s}`;
}

function addDays(iso, n) {
  const d = new Date(iso + "T12:00:00Z");
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

// Probability cell, shaded on a single-hue ramp (magnitude)
function oddsCell(p) {
  const td = el("td", { class: "odds" }, fmtPct(p));
  const mix = Math.round(Math.min(p, 1) * 62);
  if (mix > 0) td.style.background = `color-mix(in oklab, var(--seq) ${mix}%, transparent)`;
  return td;
}

function teamLink(abbrev, label) {
  return el("a", { href: `team.html?t=${abbrev}` }, label ?? abbrev);
}

function renderChrome(active, asOf) {
  const links = [["index.html", "Standings odds"], ["games.html", "Games"], ["accuracy.html", "Accuracy"], ["about.html", "Method"]];
  document.querySelector(".site-header").append(
    el("div", { class: "inner" },
      el("a", { class: "brand", href: "index.html" }, "Playoff Odds"),
      el("nav", { class: "nav" }, links.map(([href, label]) =>
        el("a", { href, "aria-current": href === active ? "page" : null }, label))))
  );
  const footer = document.querySelector(".site-footer");
  if (footer) {
    footer.append(
      el("p", {}, "Unofficial fan project. Not affiliated with the NHL. Data from the public NHL API."),
      asOf ? el("p", {}, `Model updated with games through ${fmtDate(addDays(asOf, -1))}.`) : null
    );
  }
}

/**
 * Line chart (SVG). y is a 0..yMax scale.
 *  opts.dates:  ["2026-10-01", ...]
 *  opts.series: [{ id, label, color, values: [number|null], emphasis: bool }]
 *  opts.format: value -> string
 *  opts.refLine: optional y value to draw as a labeled reference
 */
function lineChart(container, opts) {
  const { dates, series, yMax = 1, format = fmtPct, height = 300, refLine = null } = opts;
  const xTick = i => opts.xLabels ? opts.xLabels[i] : fmtDate(dates[i]);
  const xTip = i => opts.xLabels ? opts.xLabels[i] : fmtDate(dates[i], true);
  container.innerHTML = "";
  container.classList.add("chart");
  const width = Math.max(container.clientWidth, 280);
  const emph = series.filter(s => s.emphasis);
  const margin = { top: 12, right: emph.length ? 78 : 16, bottom: 26, left: 38 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const n = dates.length;
  const x = i => margin.left + (n <= 1 ? w / 2 : (i / (n - 1)) * w);
  const y = v => margin.top + h - (v / yMax) * h;
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs = {}) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  };

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img",
    "aria-label": opts.ariaLabel || "Line chart" });
  const grid = svgEl("g", { class: "grid" });
  const ticksY = [0, 0.25, 0.5, 0.75, 1].map(f => f * yMax);
  for (const t of ticksY) {
    grid.append(svgEl("line", { x1: margin.left, x2: margin.left + w, y1: y(t), y2: y(t), class: t === 0 ? "base" : "" }));
    const label = svgEl("text", { x: margin.left - 6, y: y(t) + 4, "text-anchor": "end", class: "tick" });
    label.textContent = format(t);
    grid.append(label);
  }
  // X ticks: about one per 60px, snapped to dates
  const maxTicks = Math.max(2, Math.floor(w / 70));
  const step = Math.max(1, Math.ceil((n - 1) / maxTicks));
  for (let i = 0; i < n; i += step) {
    const label = svgEl("text", { x: x(i), y: margin.top + h + 18, "text-anchor": i === 0 ? "start" : "middle", class: "tick" });
    label.textContent = xTick(i);
    grid.append(label);
  }
  svg.append(grid);

  const path = values => {
    let d = "";
    values.forEach((v, i) => {
      if (v == null) return;
      d += (d && values[i - 1] != null ? "L" : "M") + x(i).toFixed(1) + "," + y(v).toFixed(1);
    });
    return d;
  };
  for (const s of series.filter(s => !s.emphasis)) svg.append(svgEl("path", { d: path(s.values), class: "ctx" }));
  for (const s of emph) svg.append(svgEl("path", { d: path(s.values), class: "series", stroke: s.color }));

  // End markers + direct labels, nudged apart so they never collide
  const ends = emph.map(s => {
    let i = s.values.length - 1;
    while (i > 0 && s.values[i] == null) i--;
    return { s, i, v: s.values[i], ty: y(s.values[i]) };
  }).sort((a, b) => a.ty - b.ty);
  for (let k = 1; k < ends.length; k++) ends[k].ty = Math.max(ends[k].ty, ends[k - 1].ty + 15);
  for (const e of ends) {
    svg.append(svgEl("circle", { cx: x(e.i), cy: y(e.v), r: 4, fill: e.s.color, stroke: cssVar("--surface"), "stroke-width": 2 }));
    const t = svgEl("text", { x: x(e.i) + 10, y: e.ty + 4, class: "end-label" });
    t.append(document.createTextNode(e.s.label + " "));
    const v = svgEl("tspan", { class: "v" });
    v.textContent = format(e.v);
    t.append(v);
    svg.append(t);
  }

  // Hover layer: crosshair + tooltip, also keyboard accessible
  const cross = svgEl("line", { class: "crosshair", y1: margin.top, y2: margin.top + h, visibility: "hidden" });
  const dots = svgEl("g");
  svg.append(cross, dots);
  const overlay = svgEl("rect", { x: margin.left, y: margin.top, width: w, height: h, class: "overlay", tabindex: 0 });
  svg.append(overlay);
  container.append(svg);
  const tip = el("div", { class: "tooltip", hidden: true });
  container.append(tip);

  const tipSeries = emph.length ? emph : series;
  let current = -1;
  const show = i => {
    current = i;
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
    dots.innerHTML = "";
    for (const s of emph) {
      if (s.values[i] == null) continue;
      dots.append(svgEl("circle", { cx: x(i), cy: y(s.values[i]), r: 4, fill: s.color, stroke: cssVar("--surface"), "stroke-width": 2 }));
    }
    const rows = tipSeries.filter(s => s.values[i] != null).sort((a, b) => b.values[i] - a.values[i]).slice(0, 8);
    tip.replaceChildren(
      el("div", { class: "t-date" }, xTip(i)),
      ...rows.map(s => el("div", { class: "t-row" },
        el("span", {}, s.emphasis ? el("i", { class: "swatch", style: `background:${s.color}` }) : null, s.label),
        el("b", {}, format(s.values[i]))))
    );
    tip.hidden = false;
    const tw = tip.offsetWidth;
    const left = x(i) + 14 + tw > width ? x(i) - 14 - tw : x(i) + 14;
    tip.style.left = left + "px";
    tip.style.top = margin.top + "px";
  };
  const hide = () => {
    cross.setAttribute("visibility", "hidden");
    dots.innerHTML = "";
    tip.hidden = true;
    current = -1;
  };
  overlay.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) * (width / r.width);
    show(Math.max(0, Math.min(n - 1, Math.round(((px - margin.left) / w) * (n - 1)))));
  });
  overlay.addEventListener("pointerleave", hide);
  overlay.addEventListener("blur", hide);
  overlay.addEventListener("keydown", ev => {
    if (ev.key === "ArrowLeft") show(Math.max(0, (current < 0 ? n : current) - 1));
    else if (ev.key === "ArrowRight") show(Math.min(n - 1, current + 1));
    else return;
    ev.preventDefault();
  });

  if (refLine != null) {
    svg.insertBefore(svgEl("line", { x1: margin.left, x2: margin.left + w, y1: y(refLine), y2: y(refLine), stroke: cssVar("--axis"), "stroke-width": 1 }), overlay);
  }
}

// Re-render charts when the viewport or color scheme changes
function onResize(fn) {
  let last = window.innerWidth;
  window.addEventListener("resize", () => {
    if (window.innerWidth !== last) { last = window.innerWidth; fn(); }
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fn);
}

function seriesColor(slot) {
  return cssVar(SERIES_VARS[slot]);
}

function storageGet(key, fallback) {
  try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}
