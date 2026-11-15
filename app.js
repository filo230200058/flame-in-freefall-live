"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const NR = "not_reported";
const compareSet = new Set();
let facetsCache = null;

async function api(path) {
  const r = await fetch(path);
  const j = await r.json().catch(() => ({ error: "Bad response from server" }));
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
function fail(el, e) { el.innerHTML = `<div class="notice abstain">Error: ${esc(e.message || e)}</div>`; }
function show(v) { return v === NR || v === "" || v == null ? `<span class="muted"><i>Not reported</i></span>` : esc(v); }
function cite(loc, url, kind) {
  const isReport = kind === "ntrs_report" || /^NTRS /.test(loc || "");
  if (kind === "psi_document") return `<div class="cite"><span class="tag warn">PSI document (plan, not results)</span> Source: ${esc(loc)}</div>`;
  const isData = kind === "psi_data_file" || (/psi_files\//.test(loc || "") && !/SRD_/.test(loc || ""));
  const u = url ? ` <a href="${esc(url)}" target="_blank" rel="noopener">open</a>` : "";
  const badge = isData ? `<span class="tag ok">PSI measured data</span>` : isReport ? `<span class="tag info">NASA report</span>` : `<span class="tag">PSI metadata</span>`;
  return `<div class="cite">${badge} Source: ${esc(loc)}${u}</div>`;
}
const pct = (x) => (x == null ? "n/a" : (x * 100).toFixed(0) + "%");
const f3 = (x) => (x == null ? "n/a" : Number(x).toFixed(3));

/* ---------------- tabs ---------------- */
const loaded = {};
function openTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + name));
  if (!loaded[name]) { loaded[name] = true; ({ map: loadMap, runs: () => loadRuns(new URLSearchParams()), overview: loadOverview, explorer: initExplorer, coverage: loadCoverage, method: loadMethod, demo: loadDemo }[name] || (() => {}))(); }
  if (name === "compare") loadCompare();
  history.replaceState(null, "", "#" + name);
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => openTab(b.dataset.tab)));

/* ---------------- overview ---------------- */
async function loadOverview() {
  const el = $("#tab-overview");
  el.innerHTML = `<p class="muted">Loading live counts…</p>`;
  try {
    const o = await api("/api/overview");
    const snap = o.catalog_snapshot || {};
    const areas = Object.entries(o.facets.research_area);
    const maxA = Math.max(...areas.map((a) => a[1]));
    el.innerHTML = `
      <h2>How the evidence flows</h2>
      <div class="card" style="overflow-x:auto">${flowchart(o)}</div>
      <h2>What is indexed right now</h2>
      <div class="grid">
        <div class="stat"><b>${o.n_records}</b><span>PSI investigations indexed (official catalog snapshot)</span></div>
        <div class="stat"><b>${o.n_combustion}</b><span>in the Combustion Science research area</span></div>
        <div class="stat"><b>${o.n_fire_safety_subarea}</b><span>combustion records with a fire-related sub-area</span></div>
        <div class="stat"><b>${o.n_local_archives}</b><span>also present as local ZIP archives (metadata only)</span></div>
        <div class="stat"><b>${o.n_chunks.toLocaleString()}</b><span>cited evidence sentences</span></div>
        <div class="stat"><b>${o.n_reports}</b><span>public NASA (NTRS) report links on ${o.n_records_with_reports} investigations</span></div>
        <div class="stat"><b>${o.n_report_chunks.toLocaleString()}</b><span>cited sentences from those reports (page-level)</span></div>
        <div class="stat"><b>${o.n_runs}</b><span>measured test runs from ${o.n_run_tables} PSI experimental tables</span></div>
        <div class="stat"><b>${o.n_data_files}</b><span>PSI data files parsed (on ${o.n_with_measurements_inspected} investigation${o.n_with_measurements_inspected === 1 ? "" : "s"}; supplied by the owner)</span></div>
      </div>
      <p class="hint">Catalog snapshot: <code>${esc(snap.url || "n/a")}</code>, retrieved ${esc(snap.accessed_utc || "n/a")} (reported total ${esc(snap.reported_total)}). Every count on this page is computed from the indexed records when the page loads.</p>
      <div class="explorer" style="grid-template-columns:1fr 1fr">
        <div class="card"><h4>Research areas (official PSI field)</h4>
          ${areas.map(([k, v]) => `<div class="small">${esc(k)} <span class="muted">(${v})</span></div><div class="bar"><i style="width:${(100 * v) / maxA}%"></i></div>`).join("")}
        </div>
        <div class="card"><h4>Trained model</h4>
          ${o.model.available ? `<p class="small">A TF-IDF and softmax-regression classifier, trained from scratch on ${o.model.n_training} PSI records, predicts each investigation's official research area.
            Grouped cross-validation: <b>${pct(o.model.cv_relevance_accuracy?.mean)}</b> accuracy at telling fire-safety (combustion) investigations from the rest (baseline 82%). Its full 6-area accuracy is <b>${f3(o.model.cv_accuracy?.mean)}</b> and macro-F1 <b>${f3(o.model.cv_macro_f1?.mean)}</b>.</p>
            <p class="small muted">In search, it classifies your query and gives a small boost to records in the predicted area. It does not predict fire risk.</p>
            <button class="ghost small" onclick="openTab('method')">See evaluation</button>` : `<p class="notice">Model artifact missing. Run <code>python ml/train.py</code>.</p>`}
        </div>
      </div>
      <h3>Local archives (read-only originals)</h3>
      <div class="tablewrap"><table><tr><th>Archive</th><th>Internal ID</th><th>Official acronym</th><th>Mapping</th><th>SHA-256</th></tr>
        ${o.local_archives.map((a) => `<tr><td class="small">${esc(a.file)}</td><td>${esc(a.id)}</td><td>${esc(a.acronym)}</td>
          <td class="small"><span class="tag ${a.mapping_status === "confirmed" ? "ok" : "warn"}">${esc(a.mapping_status)}</span><br>${esc(a.mapping_note || "")}</td>
          <td class="cite">${esc((a.sha256 || "").slice(0, 16))}…</td></tr>`).join("")}
      </table></div>
      <p class="hint">Each archive holds one ISA-Tab <code>i_Investigation.txt</code> and a 56-byte placeholder sample sheet. None contain experimental tables, sensor data or reports. Conflicts between the archives and the catalog: ${esc(o.conflicts)}.</p>`;
  } catch (e) { fail(el, e); }
}

/* ---------------- charts (inline SVG, no libraries) ---------------- */
function svgText(x, y, s, opt = {}) {
  return `<text x="${x}" y="${y}" fill="${opt.fill || "var(--text)"}" font-size="${opt.size || 13}" font-weight="${opt.weight || 400}" text-anchor="${opt.anchor || "middle"}" font-family="system-ui, sans-serif">${esc(s)}</text>`;
}
function flowchart(o) {
  const cols = [
    { title: "1 · SOURCES", x: 10, items: [[`PSI catalog`, `${o.n_records} investigations`], [`Original ZIPs`, `${o.n_local_archives} archives, hashed`], [`Experimental tables`, `${o.n_run_tables} tables`], [`PSI data files`, `${o.n_data_files} files (FLEX…)`], [`NASA reports (NTRS)`, `${o.n_reports} links`]] },
    { title: "2 · EXTRACT & CHECK", x: 290, items: [[`Keep raw values`, `missing = not reported`], [`Cross-check sources`, `${o.conflicts} conflicts`], [`Flag suspect cells`, `never auto-fixed`]] },
    { title: "3 · LINKED EVIDENCE", x: 570, items: [[`Cited sentences`, `${o.n_chunks.toLocaleString()}`], [`Measured test runs`, `${o.n_runs}`], [`Quoted outcomes`, `${o.n_outcomes}`], [`Trained classifier`, `research area`]] },
    { title: "4 · ANSWERS", x: 850, items: [[`Explorer & detail`, `explained ranking`], [`Test runs`, `measured + outcomes`], [`Assistant`, `quotes, abstains`], [`Scenario`, `gaps flagged`], [`Coverage`, `denominators`]] },
  ];
  const W = 240, H = 54, gap = 14, top = 40;
  const pos = cols.map((c) => { const total = c.items.length * H + (c.items.length - 1) * gap; const y0 = top + (5 * H + 4 * gap - total) / 2; return c.items.map((_, i) => y0 + i * (H + gap)); });
  let s = `<svg viewBox="0 0 1100 ${top + 5 * H + 4 * gap + 10}" style="width:100%;min-width:900px" role="img" aria-label="Flowchart: sources are extracted and checked, linked into evidence, and served as answers">`;
  s += `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--accent)"/></marker></defs>`;
  for (let c = 0; c < cols.length - 1; c++) {
    const xa = cols[c].x + W, xb = cols[c + 1].x;
    const ya = pos[c].reduce((a, b) => a + b, 0) / pos[c].length + H / 2, yb = pos[c + 1].reduce((a, b) => a + b, 0) / pos[c + 1].length + H / 2;
    pos[c].forEach((y) => { s += `<path d="M${xa},${y + H / 2} C${xa + 20},${y + H / 2} ${xa + 20},${ya} ${xa + 20},${ya}" stroke="var(--line)" stroke-width="1.5" fill="none"/>`; });
    s += `<line x1="${xa + 20}" y1="${ya}" x2="${xb - 4}" y2="${yb}" stroke="var(--accent)" stroke-width="2.5" marker-end="url(#ah)"/>`;
  }
  cols.forEach((c, ci) => {
    s += svgText(c.x + W / 2, 18, c.title, { fill: "var(--accent2)", size: 13, weight: 700 });
    c.items.forEach(([a, b], i) => {
      const y = pos[ci][i];
      s += `<rect x="${c.x}" y="${y}" width="${W}" height="${H}" rx="10" fill="var(--panel2)" stroke="var(--line)"/>`;
      s += svgText(c.x + W / 2, y + 22, a, { size: 14, weight: 600 }) + svgText(c.x + W / 2, y + 41, b, { fill: "var(--muted)", size: 12 });
    });
  });
  return s + `</svg><p class="small muted">Counts are live from the indexed data. Each box’s output keeps a citation back to its source.</p>`;
}
function scatterO2(runs) {
  // x = initial O2 (% vol), rows = material · geometry, dot colour = reported ignition outcome
  const rs = runs.filter((u) => u.o2_initial && u.o2_initial.status === "reported" && u.outcome);
  if (!rs.length) return "";
  const grp = (u) => `${(u.material_class || "").replace(" (per SRD manifest)", "")} · ${u.geometry}`.replace("not_reported", "unspecified");
  const groups = [...new Set(rs.map(grp))].sort();
  const vals = rs.map((u) => u.o2_initial.value), lo = Math.floor(Math.min(...vals)) - 0.5, hi = Math.ceil(Math.max(...vals)) + 0.5;
  const L = 300, R = 1060, rowH = 34, top = 20, H = top + groups.length * rowH + 50;
  const x = (v) => L + ((v - lo) / (hi - lo)) * (R - L);
  const col = { sustained: "var(--ok)", not_sustained: "var(--bad)" };
  let s = `<svg viewBox="0 0 1100 ${H}" style="width:100%;min-width:720px" role="img" aria-label="Initial oxygen of each BASS-II run by material, coloured by whether a flame was sustained">`;
  for (let t = Math.ceil(lo); t <= hi; t++) { s += `<line x1="${x(t)}" y1="${top - 6}" x2="${x(t)}" y2="${H - 40}" stroke="var(--line)" stroke-width="1"/>` + svgText(x(t), H - 22, `${t}%`, { fill: "var(--muted)", size: 12 }); }
  groups.forEach((g, i) => {
    const y = top + i * rowH + rowH / 2;
    s += svgText(L - 12, y + 4, g, { anchor: "end", size: 13 });
    const pts = rs.filter((u) => grp(u) === g);
    pts.forEach((u, k) => {
      const c = col[u.outcome.ignition] || "var(--muted)";
      const jitter = ((k * 7) % 5 - 2) * 3;
      s += `<circle cx="${x(u.o2_initial.value).toFixed(1)}" cy="${y + jitter}" r="6" fill="${c}" fill-opacity="0.85" stroke="var(--bg)" stroke-width="1"><title>${esc(u.run_id)}: ${u.o2_initial.value}% O2, ${esc(u.outcome.ignition)}</title></circle>`;
    });
  });
  s += svgText((L + R) / 2, H - 4, "Initial O2 (% by volume, calibrated)", { fill: "var(--muted)", size: 12 });
  return s + `</svg><div class="small"><span style="color:var(--ok)">●</span> sustained flame &nbsp; <span style="color:var(--bad)">●</span> no sustained flame &nbsp; <span style="color:var(--muted)">●</span> not reported / unreviewed. Hover a dot for its test ID.</div>`;
}
function barsTermination(runs) {
  const counts = {};
  runs.forEach((u) => { const t = (u.outcome || {}).termination; if (t && u.outcome.status === "reported") counts[t] = (counts[t] || 0) + 1; });
  const e = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!e.length) return "";
  const max = e[0][1];
  return `<div class="card"><h4>How the burns ended (reported outcomes)</h4>${e.map(([k, v]) => `<div class="small">${esc(k.replace(/_/g, " "))} <b>${v}</b></div><div class="bar"><i style="width:${(100 * v) / max}%"></i></div>`).join("")}</div>`;
}

/* ---------------- explorer ---------------- */
async function initExplorer() {
  try { facetsCache = facetsCache || (await api("/api/facets")); } catch (e) { return fail($("#filters"), e); }
  const F = facetsCache;
  const sel = (name, label, obj, extra = "") => `<label>${label}<select name="${name}"><option value="">any</option>${Object.entries(obj).map(([k, v]) => `<option value="${esc(k)}">${esc(k === NR ? "Not reported" : k)} (${v})</option>`).join("")}</select>${extra}</label>`;
  $("#filters").innerHTML = `
    <label>PSI ID / name<input name="psi" placeholder="e.g. PSI-26, FLEX"></label>
    ${sel("area", "Research area", F.research_area)}
    ${sel("study_type", "Study type (derived)", F.study_type)}
    ${sel("gravity", "Gravity regime tested", F.gravity_tested)}
    ${sel("material", "Material mentioned", F.materials)}
    ${sel("condition", "Condition mentioned", F.condition_mentioned)}
    ${sel("tag", "Research intent (rule tag)", F.intent_tags)}
    ${sel("hardware", "Hardware", F.hardware)}
    ${sel("platform", "Flight platform", F.platform)}
    ${sel("file_type", "Files available locally", F.file_types)}
    <label><input type="checkbox" name="model" checked style="width:auto"> Use trained model in ranking</label>
    <button type="button" class="ghost small" id="clearFilters">Clear filters</button>`;
  $("#filters").addEventListener("change", runSearch);
  $("#clearFilters").addEventListener("click", () => { $("#filters").querySelectorAll("select,input[name=psi]").forEach((s) => (s.value = "")); runSearch(); });
  runSearch();
}
$("#searchForm").addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });

async function runSearch() {
  if (!facetsCache) return;
  const p = new URLSearchParams({ q: $("#q").value, limit: "40" });
  $("#filters").querySelectorAll("select,input[name=psi]").forEach((s) => s.value && p.set(s.name, s.value));
  p.set("model", $("#filters input[name=model]").checked ? "1" : "0");
  const out = $("#results");
  out.innerHTML = `<p class="muted">Searching…</p>`;
  try {
    const r = await api("/api/search?" + p);
    const qc = r.query_classification;
    $("#searchMeta").innerHTML = `${r.n_results} matching record(s). <span title="${esc(r.ranking_formula)}">Ranking: ${esc(r.ranking_formula)}</span>
      ${qc && qc.probs ? `<br>Model reading of your query: ${Object.entries(qc.probs).slice(0, 3).map(([k, v]) => `${esc(k)} ${pct(v)}`).join(", ")}` : qc ? `<br>${esc(qc.note || "")}` : ""}`;
    if (!r.results.length) {
      out.innerHTML = `<div class="notice">No indexed PSI record matches this query and filter set. Try fewer words or clear the filters. The catalog only covers PSI investigation metadata.</div>`;
      return;
    }
    out.innerHTML = r.results.map(resultCard).join("");
    bindResultButtons(out);
  } catch (e) { fail(out, e); }
}

function resultCard(c) {
  const x = c.explanation;
  const why = x.matched_terms.length
    ? `<div class="why"><b>Why it matched:</b> query terms found: ${x.matched_terms.map((t) => `<span class="tag ok">${esc(t)}</span>`).join("")}
       ${x.query_terms_not_found.length ? ` not found: ${x.query_terms_not_found.map((t) => `<span class="tag bad">${esc(t)}</span>`).join("")}` : ""}
       <div class="small muted">Evidence signal (BM25 on source text): ${f3(x.lexical_normalized)}${x.model_p_area_given_query != null ? `. Model output, P(${esc(c.research_area)} | query): ${f3(x.model_p_area_given_query)} with weight ${x.model_weight}` : ""}</div></div>`
    : `<div class="why small muted">Listed because it passes the selected filters (no text query).</div>`;
  const ev = c.evidence.map((e) => `<div class="ev">${esc(e.text)}${cite(e.location, e.url, e.source_kind)}</div>`).join("");
  const missing = c.missing_dimensions.length ? `<div class="small muted">Not stated in metadata: ${c.missing_dimensions.map((m) => esc(m.replace(/_/g, " "))).join(", ")}</div>` : "";
  return `<div class="card result">
    <div class="head"><div><h4>${esc(c.id)} · ${c.acronym !== NR ? esc(c.acronym) + " · " : ""}${esc(c.title)}</h4>
      <span class="tag area">${esc(c.research_area)}</span><span class="tag">${esc(c.study_type)}</span><span class="tag">${esc(c.gravity_tested)}</span>
      ${c.has_local_archive ? `<span class="tag ok">local ZIP</span>` : ""}${c.intent_tags.map((t) => `<span class="tag warn" title="rule-derived tag">${esc(t.replace(/_/g, " "))}</span>`).join("")}</div>
      <div class="score" title="relevance score, not risk">${c.score ? c.score.toFixed(3) : ""}</div></div>
    ${why}${ev}${missing}
    <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
      <button class="ghost small" data-detail="${esc(c.id)}">Details &amp; citations</button>
      <button class="ghost small" data-compare="${esc(c.id)}">${compareSet.has(c.id) ? "Remove from compare" : "Add to compare"}</button>
    </div></div>`;
}
function bindResultButtons(root) {
  root.querySelectorAll("[data-detail]").forEach((b) => b.addEventListener("click", () => openDetail(b.dataset.detail)));
  root.querySelectorAll("[data-compare]").forEach((b) => b.addEventListener("click", () => {
    const id = b.dataset.compare;
    compareSet.has(id) ? compareSet.delete(id) : compareSet.add(id);
    b.textContent = compareSet.has(id) ? "Remove from compare" : "Add to compare";
    $("#cmpCount").textContent = compareSet.size;
  }));
}

/* ---------------- detail ---------------- */
async function openDetail(id) {
  const body = $("#drawerBody");
  $("#drawer").classList.add("open");
  body.innerHTML = `<p class="muted">Loading ${esc(id)}…</p>`;
  try {
    const r = await api("/api/record/" + encodeURIComponent(id));
    const F = r.fields, d = r.derived;
    const fld = (k, label) => {
      const f = F[k]; if (!f) return "";
      const conflict = f.status === "conflicting" ? `<div class="notice">Conflicting value in local archive (${esc(f.conflict.local_location)}): ${esc(f.conflict.local_zip_value)}</div>` : "";
      return `<h3>${label}</h3>${f.status === NR ? `<p class="muted"><i>Not reported</i> in the source</p>` : `<p style="white-space:pre-wrap">${esc(f.value)}</p>`}${conflict}${cite(f.source.location, f.source.url)}`;
    };
    const condRows = Object.entries(d.conditions).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.status === "not_extracted" ? `<span class="muted">Not reported / not extracted</span>` :
      v.mentions.slice(0, 3).map((m) => `<div class="ev">${esc(m.sentence)}${m.values.length ? `<br><b>Value mentioned:</b> ${m.values.map((x) => esc(x.text)).join(", ")}` : ""}${cite(m.location, r.url)}</div>`).join("")}</td></tr>`).join("");
    const gm = Object.entries(d.gravity_mentions).filter(([, v]) => v.length).map(([k, v]) => `<div class="ev"><b>${esc(k.replace(/_/g, " "))}</b>: ${esc(v[0].sentence)}${cite(v[0].location, r.url)}</div>`).join("") || `<p class="muted">No gravity terms in the narrative.</p>`;
    const mc = r.model_check;
    body.innerHTML = `
      <h2>${esc(r.id)} · ${esc(F.acronym?.value !== NR ? F.acronym.value : "")} ${esc(F.title.value)}</h2>
      <p><a href="${esc(r.url)}" target="_blank" rel="noopener">Open the official PSI record</a> · leakage group ${esc(r.group)}</p>
      <div class="notice small">This page shows PSI <b>metadata</b> (what the investigation set out to do). Measured results are not included unless a data file was inspected. Data files: ${esc(r.file_types.note || r.file_types.status)}</div>
      <div class="kv">
        <div>Research area</div><div>${show(F.research_area.value)}</div>
        <div>Sub-area</div><div>${show(F.subresearch_area.value)}</div>
        <div>Project type</div><div>${show(F.project_type.value)}</div>
        <div>Flight platform</div><div>${show(F.flight_platform.value)}</div>
        <div>Hardware</div><div>${show(F.hardware.value)}${F.hardware_link.value !== NR ? ` · <a href="${esc(F.hardware_link.value)}" target="_blank" rel="noopener">link</a>` : ""}</div>
        <div>Dates</div><div>${show(F.start_date.value)} to ${show(F.end_date.value)}</div>
        <div>Managing center</div><div>${show(F.managing_center.value)}</div>
        <div>Study type (derived)</div><div>${esc(d.study_type)}</div>
        <div>Gravity regime tested (derived)</div><div>${esc(d.gravity_tested.value)} <span class="muted small">from ${esc(d.gravity_tested.basis)}</span></div>
        <div>Keywords</div><div>${r.keywords.map((k) => `<span class="tag">${esc(k)}</span>`).join("") || show("")}</div>
        <div>Related</div><div>${r.related.map((k) => `<a href="#" data-detail="${esc(k)}">${esc(k)}</a>`).join(", ") || show("")}</div>
        ${mc ? `<div>Model check (held out)</div><div class="small">Out-of-fold prediction: <b>${esc(mc.predicted)}</b> (official: ${esc(mc.label)}). P(Combustion)=${f3(mc.p_combustion)}. ${mc.predicted === mc.label ? "Agrees." : "<span class='status-different_regime'>Disagrees with the official label.</span>"}</div>` : ""}
      </div>
      ${fld("objectives", "Objective")}${fld("approach", "Approach")}${fld("hypothesis", "Hypothesis")}${fld("research_impacts", "Research impacts")}
      <h3>Conditions mentioned in the narrative (not measurements)</h3>
      <div class="tablewrap"><table><tr><th>Dimension</th><th>Evidence</th></tr>${condRows}</table></div>
      <h3>Gravity terms in the narrative</h3>${gm}
      <h3>Materials mentioned</h3>${d.materials.length ? d.materials.map((m) => `<div class="ev"><b>${esc(m.material)}</b>: ${esc(m.sentence)}${cite(m.location, r.url)}</div>`).join("") : `<p class="muted">None extracted.</p>`}
      <h3>Research-intent tags (keyword rules, then reviewed)</h3>${d.intent_tags.length ? d.intent_tags.map((t) => `<div class="ev"><b>${esc(t.tag.replace(/_/g, " "))}</b> ${t.review && t.review.verdict === "partial" ? `<span class="tag warn">context only</span>` : ""}<span class="small muted">review: ${esc(t.review?.verdict || "unreviewed")}${t.review?.reason ? " (" + esc(t.review.reason) + ")" : ""}</span><br>e.g. "${esc(t.evidence[0].sentence)}"${cite(t.evidence[0].location, r.url)}</div>`).join("") + `<p class="small muted">Reviewed by an AI agent at the owner's request, not by a human. Tags judged incorrect are hidden.</p>` : `<p class="muted">Not tagged (tags are only applied to Combustion Science records).</p>`}
      <h3>Sensors and measurement types (${(r.sensors || []).length})</h3>
      ${(r.sensors || []).length ? r.sensors.map((s) => `<div class="ev small"><b>${esc(s.sensor)}</b>: ${esc(s.sentence)}${cite(s.location, s.url, s.source_kind)}</div>`).join("") : `<p class="muted">None found in indexed text.</p>`}
      ${r.variables ? variablesTable(r.variables) : ""}
      ${r.data_files && r.data_files.length ? renderDataFiles(r) : ""}
      <h3>Public NASA reports linked (${r.literature.length})</h3>
      ${r.literature.length ? r.literature.map((l) => `<div class="card small"><a href="${esc(l.url)}" target="_blank" rel="noopener"><b>${esc(l.title)}</b></a> · NTRS ${esc(l.ntrs_id)} · ${esc(l.sti_type || "")} · ${esc(l.date)}
          <div class="muted">${esc((l.authors || []).slice(0, 4).join(", "))}${(l.authors || []).length > 4 ? " et al." : ""}</div>
          <div><span class="tag ${l.link_level.startsWith("investigation") ? "ok" : "warn"}">${esc(l.link_level)}</span><span class="tag">indexed: ${esc(l.text_indexed)}</span></div></div>`).join("")
        : `<p class="muted">No public NTRS report met the linking rule for this investigation.</p>`}
      ${d.publication_conditions ? `<h3>Values stated in linked reports (check the page before use)</h3><div class="tablewrap"><table><tr><th>Dimension</th><th>Sentences with values</th></tr>${Object.entries(d.publication_conditions).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.status === "not_extracted" ? `<span class="muted">Not found in linked reports</span>` :
          v.mentions.filter((m) => m.values.length).slice(0, 4).map((m) => `<div class="ev">${esc(m.sentence)}<br><b>Value stated:</b> ${m.values.map((x) => esc(x.text)).join(", ")}${cite(m.location, null, "ntrs_report")}</div>`).join("") || `<span class="muted">Mentioned without a value</span>`}</td></tr>`).join("")}</table></div>
          <p class="small muted">These are numbers as written in the report text, found by pattern matching. They are not a curated run table: check each page for which test, which units and what uncertainty.</p>` : ""}
      ${r.design_descriptors ? `<h3>Study design descriptors (local archive)</h3>${r.design_descriptors.values.map((v) => `<span class="tag">${esc(v)}</span>`).join("")}${cite(r.design_descriptors.location)}` : ""}
      <h3>Publications listed (${r.publications.length})</h3>
      ${r.publications.map((p) => `<div class="small" style="margin-bottom:6px">${esc(p.title || "(untitled)")}${p.authors ? ` · <span class="muted">${esc(p.authors)}</span>` : ""}${p.doi && p.doi !== NR && p.doi !== "not_extracted" ? ` · DOI <a href="https://doi.org/${esc(p.doi)}" target="_blank" rel="noopener">${esc(p.doi)}</a>` : ""}</div>`).join("") || show("")}
      ${r.people.length ? `<h3>People (local archive)</h3>${r.people.map((p) => `<div class="small">${esc(p.name)} · ${esc(p.role)} · ${esc(p.affiliation)}</div>`).join("")}` : ""}
      <h3>Missing fields</h3><p class="small">${r.missing_fields.length ? r.missing_fields.map((m) => `<span class="tag bad">${esc(m)}: not reported</span>`).join("") : "None"}</p>
      <h3>Sources</h3>${r.sources.map((s) => `<div class="cite">${esc(s.kind)}: ${esc(s.file || s.url)} ${s.accessed_utc ? "(accessed " + esc(s.accessed_utc) + ")" : ""}</div>`).join("")}
      <p><button class="ghost small" data-compare="${esc(r.id)}">${compareSet.has(r.id) ? "Remove from compare" : "Add to compare"}</button></p>`;
    bindResultButtons(body);
    body.querySelectorAll("a[data-detail]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); openDetail(a.dataset.detail); }));
  } catch (e) { fail(body, e); }
}
function sparkline(series, names, unit) {
  const W = 560, H = 140, P = 30, t = series.t_s;
  const vals = names.flatMap((n) => series.channels[n] || []).filter((v) => v != null);
  if (!vals.length || t.length < 2) return "";
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.15; lo -= pad; hi += pad;
  const x = (v) => P + ((v - t[0]) / (t[t.length - 1] - t[0])) * (W - P - 8);
  const y = (v) => H - 20 - ((v - lo) / (hi - lo)) * (H - 30);
  const colors = ["var(--accent)", "var(--info)", "var(--ok)", "var(--warn)"];
  const lines = names.map((n, i) => `<polyline fill="none" stroke="${colors[i % 4]}" stroke-width="1.5" points="${(series.channels[n] || []).map((v, k) => v == null ? "" : `${x(t[k]).toFixed(1)},${y(v).toFixed(1)}`).join(" ")}"/>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px;background:var(--panel2);border-radius:8px" role="img" aria-label="time series ${esc(unit)}">
    <text x="4" y="12" fill="var(--muted)" font-size="10">${hi.toFixed(2)} ${esc(unit)}</text><text x="4" y="${H - 22}" fill="var(--muted)" font-size="10">${lo.toFixed(2)}</text>
    <text x="${P}" y="${H - 4}" fill="var(--muted)" font-size="10">0 s</text><text x="${W - 50}" y="${H - 4}" fill="var(--muted)" font-size="10">${t[t.length - 1]} s</text>
    ${lines}</svg><div class="small">${names.map((n, i) => `<span style="color:${colors[i % 4]}">■</span> ${esc(n)}`).join(" &nbsp; ")}</div>`;
}
function renderDataFiles(r) {
  return `<h3>Measured data (PSI data files, ${r.data_files.length})</h3>
    <div class="notice small">Files from the PSI repository for this investigation, supplied by the project owner. Chamber readings are <b>measured</b> values. The test settings (fuel, oxygen, diluent) for each test ID are <b>not reported</b> here, because they need the PSI "Experimental table", which has not been provided.</div>
    ${r.data_files.map((f) => f.kind.startsWith("test summary") ? `<div class="card small"><b>${esc(f.file)}</b> · ${esc(f.kind)}
        <div class="tablewrap"><table>${Object.entries(f.run).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="${v === "not_reported" ? "missing" : ""}">${esc(v === "not_reported" ? "Not reported" : v)}${k === "pressure_kPa_derived" ? " <span class='muted'>(unit conversion)</span>" : ""}</td></tr>`).join("")}</table></div>
        ${f.Y_sum != null ? `<div class="muted">Sum of printed Y values: ${f.Y_sum}</div>` : ""}
        ${f.label_notes.map((n) => `<div class="muted">${esc(n)}</div>`).join("")}${cite(f.path + ", page 1 (sha256 " + f.sha256.slice(0, 12) + "…)", null, "psi_data_file")}</div>`
      : f.kind.startsWith("flame image") ? `<div class="card small"><b>${esc(f.file)}</b> · test ${esc(f.test_id)} · ${esc(f.kind)}<br>
        ${f.rows} time steps over ${f.duration_s} s (t = ${f.time_span_s[0]} to ${f.time_span_s[1]} s)${f.duplicate_timestamps ? `; ${f.duplicate_timestamps} rows with a repeated timestamp were excluded from the statistics` : ""}.
        <div class="tablewrap"><table><tr><th>Channel</th><th>Unit</th><th>First</th><th>Max (at t)</th><th>Last non-zero</th></tr>
        ${f.channels.map((c) => `<tr><td>${esc(c.channel)}</td><td>${esc(c.unit)}</td><td>${c.first}</td><td>${c.max} (${c.time_of_max_s} s)</td><td>${c.last_nonzero}</td></tr>`).join("")}</table></div>
        ${sparkline(f.series, [Object.keys(f.series.channels)[0]], f.channels[0].unit)}${sparkline(f.series, [Object.keys(f.series.channels)[1]], f.channels[1].unit)}
        ${f.notes.filter(Boolean).map((n) => `<div class="muted">${esc(n)}</div>`).join("")}${cite(f.path + " (sha256 " + f.sha256.slice(0, 12) + "…)", null, "psi_data_file")}</div>`
      : f.kind.startsWith("chamber") ? `<div class="card small"><b>${esc(f.file)}</b> · test ${esc(f.test_id)} · ${esc(f.start)} to ${esc(f.end)} (${f.duration_s} s, every ${f.sample_interval_s} s)
        <div class="tablewrap"><table><tr><th>Channel</th><th>Unit</th><th>Min</th><th>Mean</th><th>Max</th><th>SD</th><th>n</th></tr>
        ${f.channels.map((c) => `<tr><td>${esc(c.channel)}</td><td>${esc(c.unit)}</td><td>${c.min}</td><td>${c.mean}${c.mean_kpa_derived ? ` <span class="muted">(${c.mean_kpa_derived} kPa)</span>` : ""}</td><td>${c.max}</td><td>${c.sd}</td><td>${c.n}</td></tr>`).join("")}</table></div>
        ${sparkline(f.series, Object.keys(f.series.channels).filter((n) => /PSIA/i.test(n)), "psia")}
        ${sparkline(f.series, Object.keys(f.series.channels).filter((n) => /Deg/i.test(n)), "°C")}
        ${f.notes.map((n) => `<div class="muted">${esc(n)}</div>`).join("")}${cite(f.path + " (sha256 " + f.sha256.slice(0, 12) + "…)", null, "psi_data_file")}</div>`
      : `<div class="card small"><b>${esc(f.file)}</b> · test ${esc(f.test_id)} · ${esc(f.kind)}<br>${f.frames} frames, ${esc(f.first_frame)} to ${esc(f.last_frame)} (${f.duration_s} s), about ${f.approx_frame_rate_hz} frames/s, ${f.dropped_or_long_gaps} longer gaps. The images themselves are not included.${cite(f.path, null, "psi_data_file")}</div>`).join("")}`;
}
$("#drawerClose").addEventListener("click", () => $("#drawer").classList.remove("open"));
$("#drawer").addEventListener("click", (e) => { if (e.target.id === "drawer") $("#drawer").classList.remove("open"); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#drawer").classList.remove("open"); });

/* ---------------- compare ---------------- */
async function loadCompare() {
  const el = $("#compareOut");
  if (compareSet.size < 2) {
    el.innerHTML = `<h2>Compare investigations</h2><p class="muted">Add at least two records with "Add to compare" in the Research explorer.</p>
      <p><button class="ghost small" onclick="quickCompare(['PSI-26','PSI-25'])">Example: BASS vs BASS-II</button>
      <button class="ghost small" onclick="quickCompare(['PSI-69','PSI-68','PSI-98'])">Example: FLEX, FLEX-2, SAFFIRE-I</button></p>`;
    return;
  }
  el.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const c = await api("/api/compare?ids=" + [...compareSet].join(","));
    el.innerHTML = `<h2>Compare investigations</h2><p class="hint">${esc(c.note)}</p>
      <div class="tablewrap"><table><tr><th></th>${c.records.map((r) => `<th><a href="#" data-detail="${esc(r.id)}">${esc(r.id)}</a><br>${esc(r.title)}</th>`).join("")}</tr>
      ${c.rows.map((row) => `<tr><th>${esc(row.label)}</th>${row.cells.map((cell) => `<td class="${cell.missing ? "missing" : ""}">${esc(cell.value)}
        ${cell.quote ? `<div class="ev small">${esc(cell.quote)}</div>` : ""}${cell.location ? cite(cell.location, cell.url) : ""}${cell.conflict ? `<div class="notice small">Conflict: ${esc(cell.conflict.local_zip_value)}</div>` : ""}</td>`).join("")}</tr>`).join("")}
      </table></div><p><button class="ghost small" onclick="compareSet.clear();$('#cmpCount').textContent=0;loadCompare()">Clear selection</button></p>`;
    el.querySelectorAll("a[data-detail]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); openDetail(a.dataset.detail); }));
  } catch (e) { fail(el, e); }
}
function quickCompare(ids) { compareSet.clear(); ids.forEach((i) => compareSet.add(i)); $("#cmpCount").textContent = compareSet.size; openTab("compare"); }

/* ---------------- assistant ---------------- */
$("#askForm").addEventListener("submit", (e) => { e.preventDefault(); runAsk($("#askq").value); });
async function runAsk(q) {
  const el = $("#answer");
  el.innerHTML = `<p class="muted">Retrieving evidence…</p>`;
  try {
    const a = await api("/api/ask?q=" + encodeURIComponent(q));
    if (a.mode === "profile") { el.innerHTML = conclusionBox(a.conclusion) + renderProfile(a); bindDetailLinks(el); return; }
    if (a.mode === "comparison") { el.innerHTML = conclusionBox(a.conclusion) + renderComparison(a); bindDetailLinks(el); return; }
    if (a.status === "abstain") { el.innerHTML = conclusionBox({ level: "unsupported", headline: "Insufficient evidence: the assistant does not answer this.", points: [a.reason, "Try naming an investigation (e.g. BASS-II, FLEX) or using terms that appear in PSI records."], method: "Abstention rule: fewer than half of the question’s terms are covered by indexed sentences." }); return; }
    el.innerHTML = conclusionBox(a.conclusion) + renderAnswer(a);
    bindDetailLinks(el);
  } catch (e) { fail(el, e); }
}

function conclusionBox(c) {
  if (!c) return "";
  const col = { supported: "var(--ok)", partial: "var(--warn)", unsupported: "var(--bad)" }[c.level] || "var(--line)";
  const label = { supported: "SUPPORTED", partial: "PARTLY SUPPORTED", unsupported: "NOT SUPPORTED" }[c.level] || "";
  return `<div class="card" style="border:2px solid ${col}">
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><h3 style="margin:0">Conclusion</h3><span class="tag" style="border-color:${col};color:${col};font-weight:700">${label}</span></div>
    <p style="font-size:17px;margin:10px 0 6px"><b>${esc(c.headline)}</b></p>
    <ul style="margin:0">${(c.points || []).map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
    <p class="small muted" style="margin:8px 0 0">${esc(c.method || "")}</p></div>`;
}
const KIND_LABEL = { psi_metadata: "PSI metadata", ntrs_report: "NASA report", psi_data_file: "PSI measured data", psi_document: "PSI document (plan)" };
function renderAnswer(a) {
  const kindOf = (f) => (["psi_api", "local_zip", undefined, null].includes(f.source_kind) ? "psi_metadata" : f.source_kind);
  const kinds = {}, byRec = {};
  a.facts.forEach((f, i) => { f._n = i + 1; kinds[kindOf(f)] = (kinds[kindOf(f)] || 0) + 1; (byRec[f.record_id] = byRec[f.record_id] || []).push(f); });
  const recs = Object.keys(byRec);
  const statusTag = a.status === "answered" ? `<span class="tag ok">answered from sources</span>` : `<span class="tag warn">partial answer</span>`;
  const head = `<div class="card">
      <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">${statusTag}
        <span class="tag">${a.facts.length} quoted facts</span><span class="tag">${recs.length} investigation${recs.length === 1 ? "" : "s"}</span>
        ${Object.entries(kinds).map(([k, v]) => `<span class="tag info">${esc(KIND_LABEL[k] || k)}: ${v}</span>`).join("")}</div>
      <div class="grid" style="margin-top:12px">${recs.map((rid) => { const f = byRec[rid][0]; return `<div class="stat" style="padding:10px"><b style="font-size:18px"><a href="#" data-detail="${esc(rid)}">${esc(f.acronym !== NR ? f.acronym : rid)}</a></b><span>${esc(rid)} · ${byRec[rid].length} fact(s) · ${esc(f.gravity_tested)}</span></div>`; }).join("")}</div>
      ${a.uncovered_terms && a.uncovered_terms.length ? `<p class="small" style="margin-bottom:0">Not covered by any quoted sentence: ${a.uncovered_terms.map((t) => `<span class="tag bad">${esc(t)}</span>`).join("")}</p>` : ""}
    </div>`;
  const body = recs.map((rid) => `<div class="card"><h4>${esc(byRec[rid][0].acronym !== NR ? byRec[rid][0].acronym + " · " : "")}${esc(rid)}</h4>
      ${byRec[rid].map((f) => `<div class="ev"><span class="tag">${f._n}</span> ${esc(f.statement)}<div class="small muted">${f.doc_title ? "report: " + esc(f.doc_title) : "field: " + esc(f.field)}</div>${cite(f.location, f.url, f.source_kind)}</div>`).join("")}</div>`).join("");
  return `${head}
    <div class="explorer" style="grid-template-columns:1fr;gap:0"><h3>Source facts, grouped by investigation</h3>${body}</div>
    <details class="card"><summary><b>Interpretation (not from sources)</b> and how this answer was made</summary><ul>${a.interpretation.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>
      <p class="small muted">Text marked “PSI metadata” or “NASA report” is quoted word for word. “PSI measured data” items are fixed-template summaries of values read from a PSI data file. ${esc(a.method)}</p></details>`;
}
function bindDetailLinks(el) {
  el.querySelectorAll("a[data-detail]").forEach((x) => x.addEventListener("click", (e) => { e.preventDefault(); openDetail(x.dataset.detail); }));
}
function factBlock(f) {
  return `<div class="ev"><span style="white-space:pre-wrap">${esc(f.statement)}</span>${cite(f.location, f.url, f.source_kind)}</div>`;
}
function renderProfile(a) {
  const p = a.profile;
  return `<div class="notice small">Your question names one investigation, so here is everything indexed about it. Each part is quoted from its source.</div>
    <h2>${esc(p.id)} · ${esc(p.acronym !== NR ? p.acronym : "")} ${esc(p.title)}</h2>
    <p class="small"><a href="#" data-detail="${esc(p.id)}">Open full detail page</a> · <a href="${esc(p.url)}" target="_blank" rel="noopener">official PSI record</a> · conducted as: ${esc(p.gravity_tested)}</p>
    <div class="kv">${p.attributes.map((x) => `<div>${esc(x.label)}</div><div class="${x.missing ? "missing" : ""}">${x.missing ? "<i>Not reported</i>" : esc(x.value)}</div>`).join("")}</div>
    ${p.sections.map((s) => `<h3>${esc(s.label)}</h3>${s.missing ? `<p class="muted"><i>Not reported in the source.</i></p>` : s.facts.map(factBlock).join("")}`).join("")}
    <h3>Related files and documents (${p.files.length})</h3>
    <div class="tablewrap"><table><tr><th>Type</th><th>Name</th><th>Note</th></tr>${p.files.map((f) => `<tr><td>${esc(f.kind)}</td><td>${f.url ? `<a href="${esc(f.url)}" target="_blank" rel="noopener">${esc(f.name)}</a>` : esc(f.name)}</td><td class="small">${esc(f.detail)}</td></tr>`).join("")}</table></div>
    ${p.publications.length ? `<h3>Publications listed in the PSI record (${p.publications.length})</h3>${p.publications.map((x) => `<div class="small">${esc(x.title || "(untitled)")}${x.doi && !["not_reported", "not_extracted"].includes(x.doi) ? ` · <a href="https://doi.org/${esc(x.doi)}" target="_blank" rel="noopener">${esc(x.doi)}</a>` : ""}</div>`).join("")}` : ""}
    <p class="hint">${esc(a.method)}</p>`;
}
function renderComparison(a) {
  const c = a.comparison, ids = c.records.map((r) => r.id), ev = a.evidence || {};
  const cell = (x) => x.status === "value stated"
    ? `${esc(x.value)}<div class="small muted">${esc(x.source)}</div><div class="ev small">${esc(x.quote)}${cite(x.location, null, x.source === "PSI metadata" ? "" : "ntrs_report")}</div>`
    : `<span class="${x.status === "not reported" ? "status-different_regime" : "muted"}">${esc(x.status)}</span>`;
  return `<div class="notice small">Your question names ${ids.length} investigations. This table is built from their indexed records. Each value links to the sentence it came from. Nothing in the table is generated.</div>
    <h3>What each investigation set out to do</h3>
    ${c.records.map((r) => `<div class="ev"><b><a href="#" data-detail="${esc(r.id)}">${esc(r.acronym !== NR ? r.acronym : r.id)}</a></b>: ${r.objective ? esc(r.objective.statement) + cite(r.objective.location, r.objective.url) : "<i>objective not reported</i>"}</div>`).join("")}
    <h3>Shared variables and gaps</h3>
    <div class="tablewrap"><table><tr><th>Dimension</th>${c.records.map((r) => `<th>${esc(r.acronym !== NR ? r.acronym : r.id)}</th>`).join("")}</tr>
      ${c.dimensions.map((d) => `<tr><th>${esc(d.dimension)}</th>${ids.map((i) => `<td class="small">${cell(d.cells[i])}</td>`).join("")}</tr>`).join("")}</table></div>
    <h3>Roles by research intent</h3><ul>${c.roles.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
    <h3>Notes (interpretation, computed from the table)</h3><ul>${c.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
    ${ev.facts && ev.facts.length ? `<h3>Supporting quotes from these investigations</h3>${ev.facts.map((f) => `<div class="ev">${esc(f.statement)}<div class="small muted">${esc(f.acronym)}${f.doc_title ? " · report: " + esc(f.doc_title) : ""}</div>${cite(f.location, f.url, f.source_kind)}</div>`).join("")}` : ""}
    <p class="hint">${esc(a.method)}</p>`;
}

/* ---------------- crew-environment presets ---------------- */
const PRESETS = {
  iss: { gravity: "microgravity", oxygen: "21", pressure: "101.3", note: "ISS / spacecraft cabin: air at about sea-level conditions (≈21% O2, 14.7 psi ≈ 101.3 kPa), microgravity." },
  moon: { gravity: "lunar", oxygen: "30", pressure: "70.3", note: "Moon Base: expected ≥30% O2 at ≤10.2 psi (≈70.3 kPa) to shorten spacesuit pre-breathe (NASA Space Apps challenge statement), lunar gravity (~0.16 g)." },
  rover: { gravity: "lunar", oxygen: "30", pressure: "70.3", note: "Pressurized lunar rover: same expected atmosphere as the Moon Base (≥30% O2, ≤10.2 psi), lunar gravity." },
  mars: { gravity: "martian", oxygen: "", pressure: "", note: "Mars surface base: cabin atmosphere not yet specified, so only gravity is set (~0.38 g). Add O2 and pressure if you have design values." },
  earth: { gravity: "normal", oxygen: "21", pressure: "101.3", note: "Earth: normal gravity, sea-level air. Useful as the 1-g bound." },
};
$("#envPreset").addEventListener("change", (e) => {
  const p = PRESETS[e.target.value], f = $("#scenarioForm");
  if (!p) { $("#presetNote").textContent = ""; return; }
  f.gravity.value = p.gravity; f.oxygen.value = p.oxygen; f.pressure.value = p.pressure;
  $("#presetNote").textContent = "Preset: " + p.note + " Edit any field before searching.";
});

/* ---------------- fire safety map ---------------- */
async function loadMap() {
  const el = $("#mapOut");
  el.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const m = await api("/api/safety-map");
    const col = (p) => `<div class="card"><h4>${esc(p.label)} <span class="tag">${p.items.length}</span></h4>
      ${p.items.map((it) => `<details class="card" style="margin-bottom:8px"><summary><b><a href="#" data-detail="${esc(it.id)}">${esc(it.acronym)}</a></b>
          ${it.context_only ? `<span class="tag warn">context only</span>` : ""}${it.n_runs ? `<span class="tag ok">${it.n_runs} measured runs</span>` : ""}${it.n_data_files ? `<span class="tag ok">${it.n_data_files} data files</span>` : ""}${it.n_reports ? `<span class="tag info">${it.n_reports} reports</span>` : ""}</summary>
          <div class="small muted">${esc(it.title)} · ${esc(it.hardware)} · ${esc(it.gravity)}</div>
          <div class="ev small">${esc(it.evidence)}${cite(it.location, it.url)}</div></details>`).join("")}</div>`;
    el.innerHTML = `<h2>Fire safety map</h2>
      <p class="hint">${esc(m.note)}</p>
      <div class="explorer" style="grid-template-columns:repeat(3,1fr);align-items:start">${m.pillars.slice(0, 3).map(col).join("")}</div>
      <details class="card"><summary><b>${esc(m.pillars[3].label)}</b> <span class="tag">${m.pillars[3].items.length}</span>: the science behind models and future controls</summary>${col(m.pillars[3])}</details>
      <h3>Hardware families (shared apparatus makes investigations comparable)</h3>
      <div class="tablewrap"><table><tr><th>Hardware</th><th>Investigations</th></tr>${m.hardware_families.map((h) => `<tr><td>${esc(h.hardware)}</td><td>${h.investigations.map(esc).join(", ")}</td></tr>`).join("")}</table></div>
      <h3>Applications on Earth (quoted from PSI “Research Impacts/Earth Benefits”)</h3>
      ${m.earth_applications.map((e) => `<div class="ev"><b><a href="#" data-detail="${esc(e.id)}">${esc(e.acronym)}</a>:</b> ${esc(e.sentence)}${cite(e.location, e.url)}</div>`).join("") || `<p class="muted">None found.</p>`}`;
    bindDetailLinks(el);
  } catch (e) { fail(el, e); }
}
function variablesTable(v) {
  if (!v) return "";
  return `<div class="card"><h4>${esc(v.acronym)}: what varied and what stayed the same (${v.n_runs} runs)</h4>
    <div class="tablewrap"><table><tr><th>Variable</th><th>Across runs</th><th>Values (count)</th></tr>
    ${v.rows.map((r) => `<tr><td>${esc(r.variable)}</td><td class="${r.status === "varied" ? "status-partial" : r.status === "constant" ? "status-match" : "muted"}"><b>${esc(r.status)}</b></td><td class="small">${esc(r.detail)}</td></tr>`).join("")}</table></div></div>`;
}

/* ---------------- test runs ---------------- */
$("#runsForm").addEventListener("submit", (e) => { e.preventDefault(); loadRuns(new URLSearchParams(new FormData(e.target))); });
function pv(f) {
  if (!f) return `<span class="muted"><i>n/a</i></span>`;
  if (f.status === "reported") return `${f.approx ? "~" : ""}${f.value}${f.range ? ` <span class="muted">(${f.range.join("→")})</span>` : ""}`;
  if (f.status === "reported_text") return esc(f.raw);
  if (f.status === "unparsed") return `<span class="status-partial" title="kept as written; not parsed">${esc(f.raw)}</span>`;
  return `<span class="muted"><i>Not reported</i></span>`;
}
async function loadRuns(p) {
  const el = $("#runsOut");
  const inv = p.get("investigation") || "";
  $("#dlCsv").href = "/api/runs.csv" + (inv ? "?investigation=" + encodeURIComponent(inv) : "");
  $("#dlCsv").textContent = inv ? `${inv} runs as CSV` : "all runs as CSV";
  $("#dlJson").href = "/api/runs.json" + (inv ? "?investigation=" + encodeURIComponent(inv) : "");
  el.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const d = await api("/api/runs?" + p);
    const s = d.summary;
    const byInv = (inv) => d.runs.filter((u) => u.investigation === inv);
    const table = (inv, cols) => {
      const rs = byInv(inv);
      if (!rs.length) return "";
      return `<h3>${esc(rs[0].acronym)} (${inv}) · ${rs.length} run(s) · ${esc(rs[0].gravity)}</h3>
        <div class="tablewrap"><table><tr><th>Run</th>${cols.map((c) => `<th>${esc(c[0])}</th>`).join("")}<th>Flags</th><th>Source</th></tr>
        ${rs.map((u) => `<tr><td><b>${esc(u.run_id)}</b></td>${cols.map((c) => `<td class="small">${c[1](u)}</td>`).join("")}
          <td class="small">${(u.flags || []).filter((f) => !f.startsWith("O2 'Derived")).map((f) => `<div class="status-partial">${esc(f)}</div>`).join("")}${(u.operator_notes_in_settings || []).map((n) => `<span class="tag">${esc(n)}</span>`).join("")}</td>
          <td class="cite">${esc(u.source.file.split("/").pop())}:${u.source.line}</td></tr>`).join("")}</table></div>`;
    };
    el.innerHTML = `<div class="notice small">${esc(d.note)}</div>
      <div class="grid"><div class="stat"><b>${s.total_runs}</b><span>test runs from ${d.tables.length} PSI experimental tables</span></div>
        <div class="stat"><b>${s.with_measured_initial_o2}</b><span>runs with a measured initial O2 (${s.o2_range ? s.o2_range.join("–") + " %" : "n/a"})</span></div>
        <div class="stat"><b>${d.n}</b><span>runs match the current filter</span></div></div>
      ${(d.variables || []).length ? `<h3>What varied and what stayed the same</h3>${d.variables.map(variablesTable).join("")}` : ""}
      ${d.outcome_summary && d.outcome_summary.length ? `<h3>Reported outcomes, BASS-II (filtered runs)</h3><div class="notice small">${esc(d.outcome_note)}</div>
        <div class="card" style="overflow-x:auto"><h4>Initial oxygen of each run, by material: did a flame sustain?</h4>${scatterO2(byInv("PSI-25"))}</div>
        ${barsTermination(byInv("PSI-25"))}
        <div class="tablewrap"><table><tr><th>Material · geometry</th><th>Runs</th><th>Sustained flame</th><th>No sustained flame</th><th>How burns ended</th><th>Lowest O2 with sustained flame</th><th>Highest O2 without sustained flame</th></tr>
        ${d.outcome_summary.map((g) => `<tr><td>${esc(g.group)}</td><td>${g.n}</td><td>${g.sustained}</td><td>${g.not_sustained}</td>
          <td class="small">${Object.entries(g.terminations).map(([k, v]) => `${esc(k.replace(/_/g, " "))}: ${v}`).join("<br>")}</td>
          <td>${g.lowest_o2_with_sustained_flame ? `${g.lowest_o2_with_sustained_flame.o2}% (${esc(g.lowest_o2_with_sustained_flame.run)})` : "n/a"}</td>
          <td class="small">${g.highest_o2_without_sustained_flame ? `${g.highest_o2_without_sustained_flame.o2}% (${esc(g.highest_o2_without_sustained_flame.run)}): "${esc(g.highest_o2_without_sustained_flame.observation)}"` : "n/a"}</td></tr>`).join("")}</table></div>` : ""}
      ${table("PSI-25", [["Date", (u) => esc(u.date)], ["Material (as written)", (u) => esc(u.material_text)], ["Class", (u) => esc(u.material_class) + (u.material_class_from_srd ? `<div class="muted" title="${esc(u.material_class_from_srd.basis)}">from ${esc(u.material_class_from_srd.location.split("/").pop())}</div>` : "")], ["Geometry", (u) => esc(u.geometry)],
        ["Thickness mm", (u) => pv(u.thickness_mm)], ["Rod Ø mm", (u) => pv(u.rod_diameter_mm)], ["O2 start %vol", (u) => pv(u.o2_initial)], ["O2 end", (u) => pv(u.o2_final)],
        ["CO2 rise %vol", (u) => typeof u.co2_rise_pct_vol === "number" ? u.co2_rise_pct_vol : `<span class="muted"><i>n/a</i></span>`], ["CO start→end ppm", (u) => `${pv(u.co_initial)} → ${pv(u.co_final)}`],
        ["Air flow cm/s (sequence)", (u) => u.airflow.status === "reported" ? `${u.airflow.sequence.join(" → ")}<div class="muted">raw: ${esc(u.airflow.spreadsheet_raw || u.airflow.raw)}</div>` : pv(u.airflow)],
        ["Config", (u) => esc(u.flow_configuration === "not_reported" ? "" : u.flow_configuration.split(" (")[0])],
        ["Reported outcome (NASA report)", (u) => { const o = u.outcome || {}; if (o.status !== "reported") return `<span class="muted"><i>Not reported</i></span>`;
            const t5 = o.table_5_1 && Array.isArray(o.table_5_1.spread_rates_mm_s) ? `<div>spread ${o.table_5_1.spread_rates_mm_s.join(", ")} mm/s at ${o.table_5_1.velocities_cm_s.join(", ")} cm/s (Table 5.1)</div>` : "";
            return `<span class="tag ${o.ignition === "sustained" ? "ok" : o.ignition === "not_sustained" ? "bad" : ""}">${esc(o.ignition.replace(/_/g, " "))}</span><span class="tag">${esc(o.termination.replace(/_/g, " "))}</span>${o.blowoff_attempted_not_achieved ? `<span class="tag warn">blowoff tried, not reached</span>` : ""}
            ${t5}<div class="ev small" style="min-width:220px">"${esc(o.observation_verbatim)}"<div class="cite">${esc(o.source.location)} <a href="${esc(o.source.url)}" target="_blank" rel="noopener">open</a></div></div>`; }],
        ["Spreadsheet check", (u) => u.cross_check ? `<span class="${u.cross_check.status === "consistent" ? "status-match" : "status-different_regime"}">${esc(u.cross_check.status)}</span>` : ""]])}
      ${byInv("PSI-25").length ? `<p class="small muted">BASS-II column meanings follow the PSI "BASS-II data FAQ": Air display = duct flow speed in cm/s (slow ~1 min response); calibrated O2 corrected with a constant daily offset from the ISS MCA reading; CO "OR" = over range (&gt;1000 ppm). Flow configuration by PI from the FAQ: Olson tests ignite the upstream end (concurrent), Fernandez-Pello the downstream end (opposed). Every row was cross-checked against the PSI test-matrix spreadsheet.</p>` : ""}
      ${table("PSI-99", [["Material", (u) => esc(u.material_text)], ["Thickness", (u) => pv(u.thickness_mm)], ["O2 %", (u) => pv(u.o2_initial)], ["Flow cm/s", (u) => pv(u.airflow)],
        ["Direction", (u) => esc(u.flow_direction)], ["Burn time s", (u) => pv(u.burn_time_s)], ["µg burn length", (u) => pv(u.ug_burn_length)], ["µg spread mm/s", (u) => pv(u.ug_spread_rate_mm_s)],
        ["1-g burn length", (u) => pv(u["1g_burn_length"])], ["1-g spread", (u) => pv(u["1g_spread_rate"])]])}
      ${byInv("PSI-99").length ? `<p class="small muted">SAFFIRE-II O2 note (from the table): "Derived from measured CO2 production and referenced to initial O2 concentration reported by ISS." The 1-g columns are separate normal-gravity comparisons, not microgravity results.</p>` : ""}
      ${table("PSI-102", [["Material", (u) => esc(u.material_text)], ["Air flow cm/s", (u) => pv(u.airflow)], ["Weight loss mg/min", (u) => pv(u.weight_loss_rate_mg_min)], ["Aging min", (u) => pv(u.aging_min)]])}
      <h3>Source tables</h3>${d.tables.map((t) => `<div class="cite">${esc(t.file)} · ${t.source_rows} rows → ${t.runs} runs · sha256 ${esc(t.sha256.slice(0, 12))}…</div>`).join("")}`;
  } catch (e) { fail(el, e); }
}

/* ---------------- scenario ---------------- */
$("#scenarioForm").addEventListener("submit", (e) => { e.preventDefault(); runScenario(new URLSearchParams(new FormData(e.target))); });
async function runScenario(p) {
  const el = $("#scenarioOut");
  el.innerHTML = `<p class="muted">Matching evidence…</p>`;
  try {
    const s = await api("/api/scenario?" + p);
    const req = Object.entries(s.requested).filter(([, v]) => v != null);
    if (!req.length) { el.innerHTML = `<div class="notice">Set at least one scenario dimension.</div>`; return; }
    const rows = s.rows.filter((r) => r.n_match || r.n_partial);
    const dims = Object.keys((s.rows[0] || {}).dimensions || {});
    const SYM = { match: ["✓", "match", "var(--ok)"], partial: ["◐", "partial", "var(--warn)"], different_regime: ["✕", "different regime", "var(--bad)"], not_reported: ["–", "not reported", "var(--muted)"], unknown: ["?", "unknown", "var(--muted)"] };
    const verdict = dims.map((d) => {
      const c = { match: 0, partial: 0, other: 0 };
      s.rows.forEach((r) => { const st = r.dimensions[d].status; c[st === "match" || st === "partial" ? st : "other"]++; });
      const supported = c.match > 0, label = supported ? "supported" : c.partial ? "only partial evidence" : "not supported";
      return `<div class="stat" style="border-color:${supported ? "var(--ok)" : c.partial ? "var(--warn)" : "var(--bad)"}"><b style="font-size:22px">${esc(d)}</b>
        <span class="${supported ? "status-match" : c.partial ? "status-partial" : "status-different_regime"}" style="font-weight:600">${label}</span>
        <span>${c.match} match · ${c.partial} partial · ${c.other} none, of ${s.rows.length} records</span></div>`;
    }).join("");
    const cell = (v) => { const [sym, lab, colr] = SYM[v.status] || ["?", v.status, "var(--muted)"]; return `<td style="text-align:center;color:${colr};font-weight:700" title="${esc(lab + ": " + v.detail)}">${sym}<div style="font-size:11px;font-weight:400">${esc(lab)}</div></td>`; };
    el.innerHTML = conclusionBox(s.conclusion) + `
      <div class="notice small">${esc(s.note)}</div>
      <p><b>Scope:</b> ${esc(s.scope)}. <b>Requested:</b> ${req.map(([k, v]) => `<span class="tag">${esc(k)} = ${esc(v)}</span>`).join("")}</p>
      <h3>Verdict per dimension</h3><div class="grid">${verdict}</div>
      ${s.unsupported_dimensions.length ? `<div class="notice abstain"><b>Not supported by any indexed record:</b> ${s.unsupported_dimensions.map(esc).join(", ")}. There is no direct evidence for these parts of the scenario.</div>` : ""}
      <h3>Evidence matrix (${rows.length} of ${s.rows.length} records touch at least one dimension)</h3>
      <div class="tablewrap"><table><tr><th>Investigation</th><th>Conducted as</th>${dims.map((d) => `<th style="text-align:center">${esc(d)}</th>`).join("")}</tr>
        ${rows.slice(0, 25).map((r) => `<tr><td><a href="#" data-detail="${esc(r.record.id)}">${esc(r.record.acronym !== NR ? r.record.acronym : r.record.id)}</a><div class="small muted">${esc(r.record.id)}</div></td>
          <td class="small">${esc(r.record.gravity_tested)}</td>${dims.map((d) => cell(r.dimensions[d])).join("")}</tr>`).join("")}</table></div>
      <p class="small muted">✓ match · ◐ partial · ✕ different gravity regime · – not reported. Hover a cell for its detail; the evidence for each row is below.</p>
      <h3>Evidence behind each row</h3>
      ${rows.slice(0, 25).map((r) => `<details class="card"><summary><b>${esc(r.record.acronym !== NR ? r.record.acronym : r.record.id)}</b> · ${esc(r.record.title)} <span class="tag ok">${r.n_match} match</span><span class="tag warn">${r.n_partial} partial</span></summary>
        ${Object.entries(r.dimensions).map(([k, v]) => `<div class="dim"><b>${esc(k)}</b>: <span class="status-${esc(v.status)}">${esc(v.status.replace(/_/g, " "))}</span>, ${esc(v.detail)}
          ${v.quote ? `<div class="ev small">${esc(v.quote)}${cite(v.location, r.record.url)}</div>` : ""}</div>`).join("")}</details>`).join("") || `<p class="muted">No record touches any requested dimension.</p>`}
      <h3>Limitations</h3><ul>${s.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`;
    el.querySelectorAll("a[data-detail]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); openDetail(a.dataset.detail); }));
  } catch (e) { fail(el, e); }
}

/* ---------------- coverage ---------------- */
async function loadCoverage(scope = "combustion") {
  const el = $("#coverageOut");
  el.innerHTML = `<p class="muted">Computing…</p>`;
  try {
    const c = await api("/api/coverage?scope=" + scope);
    const n = c.denominator;
    const barRow = (label, v) => `<div class="small">${esc(label)}: <b>${v} / ${n}</b> (${pct(v / n)})</div><div class="bar"><i style="width:${(100 * v) / n}%"></i></div>`;
    el.innerHTML = `
      <h2>Coverage and gaps</h2>
      <p>Scope: <select id="covScope"><option value="combustion">Combustion Science records</option><option value="all">All PSI records</option></select>, denominator <b>${n}</b> investigations.</p>
      <p class="hint">${esc(c.note)}</p>
      <div class="explorer" style="grid-template-columns:1fr 1fr">
        <div class="card"><h4>Run conditions mentioned in metadata</h4>
          ${["oxygen", "pressure", "airflow"].map((k) => barRow(k + " mentioned in PSI metadata", c.conditions[k].mentioned) + barRow(k + " value in PSI metadata", c.conditions[k].with_numeric_value) + barRow(k + " value in a linked NASA report", c.conditions[k].with_numeric_value_in_linked_report)).join("")}
          <p class="small muted">"Mentioned" can be as weak as a sentence naming the variable. Values are as written in the text, not a verified run table.</p></div>
        <div class="card"><h4>Measured test runs (PSI experimental tables)</h4>
          ${barRow("investigations with an experimental table", c.runs.investigations_with_run_table)}
          <div class="small">${c.runs.runs} runs; ${c.runs.runs_with_measured_o2} with measured initial O2 (${c.runs.o2_range ? c.runs.o2_range.join("–") + " %" : "n/a"}); ${c.runs.runs_with_airflow_value} with a numeric air flow; ${c.runs.runs_flagged} flagged as suspect or ambiguous.</div>
          <p class="small muted">No run in these tables exceeds ${c.runs.o2_range ? c.runs.o2_range[1] : "?"}% O2, so higher-oxygen habitat atmospheres (e.g. ~30%) have no direct measured run here.</p></div>
        <div class="card"><h4>Public NASA reports (NTRS)</h4>
          ${barRow("investigations with a linked report", c.literature.records_with_linked_reports)}${barRow("with report full text indexed", c.literature.records_with_report_full_text)}
          <p class="small muted">${c.literature.reports_linked} report links in scope. Links use a conservative acronym/title rule; some are program-level (marked in the detail view).</p></div>
        <div class="card"><h4>Data types</h4>
          ${barRow("metadata narrative", c.data_types.metadata_narrative)}${barRow("local archive inspected", c.data_types.local_archive_inspected)}
          ${barRow("measured data inspected", c.data_types.measurements_inspected)}${barRow("data files not inspected", c.data_types.data_files_not_inspected)}</div>
        <div class="card"><h4>Gravity regime the investigation was conducted in</h4>
          ${Object.entries(c.gravity_tested).map(([k, v]) => barRow(k, v)).join("")}
          <h4>Gravity regimes mentioned in text</h4>${["microgravity", "normal_gravity", "lunar_gravity", "martian_gravity", "partial_gravity_generic"].map((k) => barRow(k.replace(/_/g, " "), c.gravity_mentioned_in_text[k] || 0)).join("")}
          <p class="small muted">A mention is not a test. For example, lunar gravity may only appear as motivation.</p></div>
        <div class="card"><h4>Field completeness</h4>${Object.entries(c.field_completeness).map(([k, v]) => barRow(k.replace(/_/g, " "), v)).join("")}
          <h4>Materials mentioned</h4>${Object.entries(c.materials).slice(0, 15).map(([k, v]) => `<span class="tag">${esc(k)} (${v})</span>`).join("") || "none"}</div>
      </div>`;
    $("#covScope").value = scope;
    $("#covScope").addEventListener("change", (e) => loadCoverage(e.target.value));
  } catch (e) { fail(el, e); }
}

/* ---------------- methodology ---------------- */
async function loadMethod() {
  const el = $("#methodOut");
  try {
    const m = await api("/api/model");
    const M = m.metrics, R = M?.results || {}, D = M?.deployed_model_detail, RE = m.ranking_eval;
    const row = (name, s) => s ? `<tr><td>${esc(name)}</td><td><b>${f3(s.combustion_accuracy?.mean)}</b></td><td>${f3(s.combustion_f1?.mean)}</td><td>${f3(s.combustion_auc?.mean)}</td><td>${f3(s.accuracy.mean)} ± ${f3(s.accuracy.sd_across_seeds)}</td><td>${f3(s.macro_f1.mean)} ± ${f3(s.macro_f1.sd_across_seeds)}</td></tr>` : "";
    el.innerHTML = `
      <h2>Methodology and limitations</h2>
      <div class="card"><h4>1. Acquisition</h4><ul>
        <li>Six local ZIP archives were hashed (SHA-256) and read in memory without being modified. Each holds ISA-Tab metadata and a placeholder sample sheet, and no measurements.</li>
        <li>The official PSI catalog was retrieved from the public endpoint the PSI website itself uses (<code>/geode-py/ws/repo/search</code>). The query, the time and the raw response are saved in <code>data/raw/psi_api/</code>.</li>
        <li>Public NASA Technical Reports (NTRS API) were searched for each combustion investigation. A report is linked only when its title or abstract names the investigation's acronym or title together with a combustion term. Linked PDFs are text-indexed with page numbers, and each link is marked investigation-specific or family/program-level.</li>
        <li>ZIP-to-record mapping comes from each archive's internal <code>Study Identifier</code>, checked against the catalog. The second PSI number in each filename belongs to a different investigation, so filenames are not trusted.</li></ul></div>
      <div class="card"><h4>2. Extraction and normalization</h4><ul>
        <li>Raw field values are kept alongside normalized fields. Every field records its source (PSI record and field name, or ZIP member and line range).</li>
        <li>Empty fields are marked <code>not_reported</code>. Conditions not found by the rules are marked <code>not_extracted</code>. Nothing is set to zero.</li>
        <li>Gravity, oxygen, pressure, airflow, materials and intent tags come from transparent keyword and regex rules (<code>pipeline/extract.py</code>). They are narrative mentions, not measurements.</li></ul></div>
      <div class="card"><h4>3. Trained model</h4>
        ${m.info ? `<p class="small"><b>${esc(m.info.name)}</b> v${esc(m.info.version)}, trained ${esc(m.info.trained_utc)}. Pretrained components: ${esc(m.info.pretrained_components)}.
          Dataset: ${m.info.dataset.n} investigations, ${m.info.dataset.n_groups} leakage groups, sha256 ${esc(m.info.dataset.sha256.slice(0, 16))}…</p>` : ""}
        <p class="small"><b>Task:</b> predict an investigation's official PSI Research Area (6 classes) from its narrative metadata. Labels are the official catalog field, so no manual or LLM labels are involved. The research-area fields themselves are excluded from the input.</p>
        <p class="small"><b>Evaluation:</b> 5-fold cross-validation, grouped so that linked investigations (the official "Related Investigations" field, plus shared acronym families such as FLEX/FLEX-2) never cross folds, repeated over ${M?.cv?.seeds?.length} seeds.</p>
        <p class="small"><b>Two readings of the same model.</b> The app uses it to tell fire-safety-relevant (Combustion Science) investigations from the rest, and it also classifies all 6 PSI research areas. The deployed model is trained on the 133 PSI records plus NASA Task Book Physical Sciences projects, which carry NASA’s own discipline labels. It is <b>evaluated only on PSI records</b>, with grouped folds. In each fold, Task Book projects that name a held-out investigation, share ≥50% of its title words or nearly duplicate its text are removed from training.</p>
        <div class="tablewrap"><table><tr><th>Model</th><th>Fire-safety relevance accuracy</th><th>Relevance F1</th><th>Relevance AUC</th><th>6-area accuracy</th><th>6-area macro-F1</th></tr>
          ${row("Majority class (baseline)", R.majority?.summary)}${row("Multinomial Naive Bayes (baseline)", R.naive_bayes?.summary)}
          ${row("TF-IDF + softmax regression, PSI only" + (M.deployed_model === "softmax_regression" ? " (deployed)" : ""), R.softmax_regression?.summary)}
          ${R.softmax_regression_plus_taskbook ? row(`Same model + ${R.softmax_regression_plus_taskbook.external_rows} NASA Task Book projects as extra training data` + (M.deployed_model === "softmax_regression_plus_taskbook" ? " (deployed)" : ""), R.softmax_regression_plus_taskbook.summary) : ""}${row("PSI only, without keywords (ablation)", R.softmax_regression_no_keywords?.summary)}</table></div>
        ${D ? `<p class="small">Bootstrap 95% CI over investigations (seed ${D.seed}): accuracy ${esc(D.bootstrap_over_investigations.accuracy_95ci.join(" to "))}, macro-F1 ${esc(D.bootstrap_over_investigations.macro_f1_95ci.join(" to "))}.</p>
          <div class="tablewrap"><table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>n</th></tr>${Object.entries(D.per_class).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${f3(v.precision)}</td><td>${f3(v.recall)}</td><td>${f3(v.f1)}</td><td>${v.support}</td></tr>`).join("")}</table></div>` : ""}
        <p class="small muted">Read the relevance figure against its baseline: always answering “not combustion” already scores 82.0%, because 24 of 133 investigations are combustion. Trained on PSI alone, the model scored 98.0% relevance (2 errors: DAFT and SAME data modelling) and 85.2% on 6 areas, and no PSI-only variant passed about 87%. Adding NASA Task Book projects fixed both relevance errors and lifted the 6-area accuracy. The remaining errors are mostly Biophysics (only 5 Task Book examples) and Fluid Physics versus Complex Fluids. This model confirms topical relevance; it adds no scientific judgment.</p></div>
      <div class="card"><h4>4. Ranking</h4>
        <p class="small">score = 0.7 × normalized BM25 over the source text + 0.3 × model P(record's official area | query). The weight was fixed in advance and not tuned on evaluation queries.</p>
        ${RE ? `<p class="small"><b>Proxy evaluation</b> (${RE.n_queries} queries built from official PSI keywords; the keyword field is removed from the index, and the model is retrained without the relevant records for each query):</p>
          <div class="tablewrap"><table><tr><th>Metric</th><th>BM25 only</th><th>BM25 + model</th><th>Difference (95% CI)</th><th>Better / worse queries</th></tr>
          ${Object.entries(RE.summary).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${f3(v.bm25)}</td><td>${f3(v.hybrid)}</td><td>${f3(v.diff_mean)} (${esc(v.diff_95ci.join(" to "))})</td><td>${v.queries_better} / ${v.queries_worse}</td></tr>`).join("")}</table></div>
          <p class="small muted">These are proxy relevance labels, not human judgments. Treat them as a sanity check, not proof of usefulness.</p>` : `<p class="notice small">Ranking evaluation not run yet: <code>python ml/eval_ranking.py</code></p>`}</div>
      <div class="card"><h4>5. Measured runs and outcomes, and why no fire-outcome model</h4><ul>
        <li>146 test runs come from PSI "Experimental table" files (BASS-II, SAFFIRE-II, SAME). Every cell is kept verbatim and cited by file and line. BASS-II was cross-checked against the PSI spreadsheet (129 of 129 consistent), and its columns are interpreted with the PSI data FAQ.</li>
        <li>125 BASS-II outcomes are quoted from the public NASA report TM-20210011385 (Appendix A, Tables 5.1 and 7.1) with page numbers. The AI agent labelled them (ignition, how the burn ended); this is not a human review.</li>
        <li>We deliberately did <b>not</b> train an ignition or extinction predictor. Only 15 tests had no sustained flame, and their quoted causes are often procedural (reused sample, igniter angle, flat tip). How a burn ended mostly reflects the operator's procedure. A model would learn the test procedure, not flammability, and could be mistaken for a hazard rating.</li></ul></div>
      <div class="card"><h4>6. Known limits</h4><ul>
        <li>PSI data files are access-restricted (HTTP 403 to automated requests). Only the files the owner downloaded and supplied are parsed: three FLEX files (one chamber pressure/temperature log and two camera timing logs). Their test settings remain not reported until the PSI "Experimental table" is supplied. Other results appear only as quoted report sentences, not a curated run table.</li>
        <li>No model here predicts ignition, flammability, fire spread or material safety. The data cannot support that.</li>
        <li>Condition extraction is rule-based and incomplete. Absence means not found, not "zero".</li>
        <li>The catalog is a ${esc(M?.trained_utc?.slice(0, 10) || "")} snapshot of PSI. Records may change upstream.</li>
        <li>Stronger models would need verified run-level tables (oxygen, pressure, flow, gravity, material, outcome) from PSI data files, plus human relevance judgments for real user queries.</li></ul></div>`;
  } catch (e) { fail(el, e); }
}

/* ---------------- demo ---------------- */
function loadDemo() {
  $("#demoOut").innerHTML = `
    <h2>Three-minute demo flow</h2>
    <p class="small muted">The question we answer: <b>"What does NASA's microgravity fire research actually show, and where are the gaps?"</b> Every step uses real indexed data with a citation.</p>
    <ol>
      <li><b>Overview (15 s):</b> 133 PSI investigations, 146 measured test runs, linked NASA reports and parsed PSI data files. All counts are computed live. <button class="ghost small" onclick="openTab('overview')">Open</button></li>
      <li><b>Explorer (30 s):</b> search <i>"flame spread over fabric with opposed airflow"</i>. Each result shows why it matched and quotes its evidence, kept separate from the trained model's contribution. <button class="ghost small" onclick="demoSearch('flame spread over fabric with opposed airflow')">Run</button></li>
      <li><b>Fire safety map (20 s):</b> investigations grouped by prevention, detection and suppression, with hardware families and quoted Earth applications. <button class="ghost small" onclick="openTab('map')">Open</button></li>
      <li><b>Test runs (40 s):</b> real BASS-II runs with measured O2, CO2, CO and airflow, plus each run's outcome quoted from the NASA report with its page. The summary shows, for example, that Nomex never kept a flame (0 of 3) even at about 22% O2. <button class="ghost small" onclick="demoRuns('PSI-25','')">Open BASS-II runs</button> <button class="ghost small" onclick="demoRuns('PSI-25','Nomex')">Nomex only</button></li>
      <li><b>Measured data (20 s):</b> FLEX (PSI-69): chamber pressure log, flame diameter over time (FLEX-187), and the FLEX-263 run sheet (methanol, D0 2.668 mm, extinction at 1.421 mm). <button class="ghost small" onclick="openDetail('PSI-69')">Open FLEX</button></li>
      <li><b>Evidence assistant (30 s):</b> <i>"What was the extinction diameter in FLEX-263?"</i> answers from the data file. Naming several studies (<i>BASS, SAFFIRE, SAME and FLEX</i>) gives a cited comparison of shared variables and gaps. <button class="ghost small" onclick="demoAsk('What was the extinction diameter in FLEX-263?')">Run</button> <button class="ghost small" onclick="demoAsk('How do BASS, SAFFIRE, SAME and FLEX relate? Identify shared variables and gaps.')">Run</button></li>
      <li><b>Scenario, the key moment (30 s):</b> the <i>Moon Base</i> preset (≥30% O2 at 10.2 psi, lunar gravity) with Nomex. The conclusion is honest: not supported. No measured run went above 22.2% O2 and none was conducted in lunar gravity. The closest high-O2 evidence is ACME BRE (tested at 10.2 psia/26.5% and 8.2 psia/34%), and micro- plus normal-gravity results can only bracket partial gravity. Then try 17% O2, which matches 86 measured BASS-II runs. <button class="ghost small" onclick="demoScenario()">Run lunar 30%</button> <button class="ghost small" onclick="demoScenario17()">Run 17%</button></li>
      <li><b>Coverage &amp; Methodology (15 s):</b> denominators for every dimension, model metrics against baselines (including the ranking test where the model gave no gain), and why we did not train a fire-outcome predictor. <button class="ghost small" onclick="openTab('coverage')">Coverage</button> <button class="ghost small" onclick="openTab('method')">Methodology</button></li>
    </ol>
    <p class="hint">All demo content comes from the indexed PSI records. No synthetic fixtures are shown in the app. Test fixtures live only in <code>tests/</code>.</p>`;
}
function demoRuns(inv, mat) {
  openTab("runs");
  const f = $("#runsForm");
  f.investigation.value = inv; f.material.value = mat;
  loadRuns(new URLSearchParams(new FormData(f)));
}
function demoScenario17() {
  openTab("scenario");
  const f = $("#scenarioForm");
  f.gravity.value = "microgravity"; f.oxygen.value = "17"; f.material.value = "PMMA"; f.focus.value = "";
  runScenario(new URLSearchParams(new FormData(f)));
}
function demoSearch(q) { openTab("explorer"); $("#q").value = q; setTimeout(runSearch, 50); }
function demoAsk(q) { openTab("assistant"); $("#askq").value = q; runAsk(q); }
function demoScenario() {
  openTab("scenario");
  const f = $("#scenarioForm");
  f.preset.value = "moon"; f.preset.dispatchEvent(new Event("change"));
  f.material.value = "nomex"; f.focus.value = "flame spread";
  runScenario(new URLSearchParams(new FormData(f)));
}

{
  const want = (location.hash || "").slice(1);
  openTab(document.getElementById("tab-" + want) ? want : "overview");
}
/* ?shot=NAME: opens a demo state for screen captures (used to make the demo video) */
(async () => {
  const shot = new URLSearchParams(location.search).get("shot");
  if (!shot) return;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const scrollTo = (sel, off = 120) => { const e = document.querySelector(sel); if (e) window.scrollTo(0, e.getBoundingClientRect().top + window.scrollY - off); };
  if (shot === "map") { openTab("map"); await wait(1200); }
  if (shot === "runs") {
    demoRuns("PSI-25", "");
    for (let i = 0; i < 40 && !document.querySelector("#runsOut svg"); i++) await wait(200);
    // headless captures do not scroll reliably: hide what sits above the chart instead
    document.querySelectorAll("#runsOut .tablewrap, #runsForm, #tab-runs > .hint").forEach((t) => (t.style.display = "none"));
    const kids = [...document.querySelector("#runsOut").children];
    const k = kids.findIndex((e) => e.tagName === "H3" && e.textContent.startsWith("Reported outcomes"));
    kids.slice(0, Math.max(k, 0)).forEach((e) => (e.style.display = "none"));
    document.querySelector(".banner").style.display = "none";
    await wait(500);
  }
  if (shot === "moon") { demoScenario(); await wait(1800); document.querySelector(".banner").style.display = "none"; }
  if (shot === "ask") { demoAsk("What was the extinction diameter in FLEX-263?"); await wait(1800); document.querySelector(".banner").style.display = "none"; }
  if (shot === "flex") { await openDetail("PSI-69"); await wait(800); const h = [...document.querySelectorAll("#drawerBody h3")].find((x) => x.textContent.startsWith("Measured")); if (h) h.scrollIntoView(); }
  document.body.dataset.shotReady = "1";
})();
