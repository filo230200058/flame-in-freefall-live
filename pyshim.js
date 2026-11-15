// Static-hosting shim: runs app/engine.py in the browser with Pyodide and answers /api/* fetches locally.
(function () {
  const FILES = [
    "app/engine.py", "ml/textml.py", "deploy/router.py",
    "data/processed/records.json", "data/processed/chunks.json", "data/processed/ingest_log.json",
    "data/processed/runs.json", "data/raw/source_manifest.json",
    "ml/artifacts/metrics.json", "ml/artifacts/ranking_eval.json", "ml/artifacts/oof_predictions.json",
    "ml/artifacts/research_area_model.json",
  ];
  const step = (t) => { const el = document.getElementById("pyLoadingStep"); if (el) el.textContent = t; };
  const ready = (async () => {
    step("Starting Python in your browser…");
    const py = await loadPyodide();
    step("Loading NASA data (" + FILES.length + " files)…");
    await Promise.all(FILES.map(async (rel) => {
      const r = await fetch("site/" + rel);
      if (!r.ok) throw new Error("missing " + rel);
      const buf = new Uint8Array(await r.arrayBuffer());
      const dir = "/site/" + rel.split("/").slice(0, -1).join("/");
      py.FS.mkdirTree(dir);
      py.FS.writeFile("/site/" + rel, buf);
    }));
    step("Building the search index and loading the model…");
    await new Promise((res) => setTimeout(res, 30));
    py.runPython("import sys; sys.path.insert(0, '/site/deploy'); import router");
    const handle = py.globals.get("router").handle;
    const el = document.getElementById("pyLoading");
    if (el) el.remove();
    return handle;
  })();
  ready.catch((e) => step("Could not start: " + e.message));

  async function answer(url) {
    const handle = await ready;
    return JSON.parse(handle(url));
  }

  const origFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const url = typeof input === "string" ? input : (input instanceof URL ? input.href : (input && input.url) || String(input));
    const u = new URL(url, location.href);
    if (u.origin === location.origin && u.pathname.includes("/api/")) {
      const [status, ctype, body] = await answer(u.pathname.replace(/^.*(\/api\/)/, "/api/") + u.search);
      return new Response(body, { status, headers: { "Content-Type": ctype } });
    }
    return origFetch(input, init);
  };

  // Download links (/api/runs.csv, /api/runs.json) are navigations, not fetches.
  document.addEventListener("click", async (e) => {
    const a = e.target.closest && e.target.closest("a");
    const href = a && a.getAttribute("href");
    if (!href || !/\/api\/runs\.(csv|json)/.test(href)) return;
    e.preventDefault();
    const [, ctype, body, name] = await answer(href);
    const blob = new Blob([body], { type: ctype });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = name || "download";
    document.body.appendChild(link);
    link.click();
    setTimeout(() => { URL.revokeObjectURL(link.href); link.remove(); }, 1000);
  }, true);
})();
