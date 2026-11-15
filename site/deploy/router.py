"""In-browser router: answers the same /api/* paths as app/server.py, using the same Engine.
Runs under Pyodide with the site files mounted at /site (see deploy/pyshim.js)."""
import json
import pathlib
import sys
import urllib.parse

sys.path.insert(0, "/site/app")
from engine import Engine  # noqa: E402

E = Engine(root=pathlib.Path("/site"))


def num(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _route(path, q):
    if path == "/api/overview":
        return 200, E.overview()
    if path == "/api/facets":
        return 200, E.facets()
    if path == "/api/search":
        filters = {k: q.get(k, "") for k in ("area", "study_type", "gravity", "platform", "hardware", "material",
                                              "tag", "condition", "file_type", "psi")}
        return 200, E.search(q.get("q", ""), filters, limit=int(q.get("limit", 25)), use_model=q.get("model", "1") != "0")
    if path.startswith("/api/record/"):
        r = E.detail(urllib.parse.unquote(path.rsplit("/", 1)[-1]))
        return (200, r) if r else (404, {"error": "unknown record"})
    if path == "/api/compare":
        ids = [i.strip() for i in q.get("ids", "").split(",") if i.strip()]
        if len(ids) < 2:
            return 400, {"error": "choose at least two records"}
        return 200, E.compare(ids)
    if path == "/api/ask":
        return 200, E.ask(q.get("q", ""))
    if path == "/api/scenario":
        return 200, E.scenario(gravity=q.get("gravity", ""), oxygen=num(q.get("oxygen")),
                               pressure_kpa=num(q.get("pressure")), airflow_cms=num(q.get("airflow")),
                               material=q.get("material", "").strip(), focus=q.get("focus", "").strip(),
                               scope=q.get("scope", "combustion"))
    if path == "/api/safety-map":
        return 200, E.safety_map()
    if path == "/api/runs":
        return 200, E.runs_query(q.get("investigation", ""), q.get("material", "").strip(),
                                 num(q.get("o2min")), num(q.get("o2max")), q.get("flagged", ""))
    if path == "/api/coverage":
        return 200, E.coverage(q.get("scope", "combustion"))
    if path == "/api/model":
        return 200, {"info": E.model_info, "metrics": E.metrics,
                     "ranking_eval": E.ranking_eval and {k: v for k, v in E.ranking_eval.items() if k != "per_query"},
                     "classify": E.classify_query(q["q"]) if q.get("q") else None}
    if path == "/api/sources":
        return 200, {"manifest": E.manifest, "ingest_log": E.ingest_log}
    return 404, {"error": "unknown endpoint"}


def handle(url):
    """Returns JSON text: [status, content_type, body_text, download_filename_or_null]."""
    u = urllib.parse.urlparse(url)
    path = u.path[u.path.find("/api/"):]
    q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
    inv = q.get("investigation", "")
    try:
        if path == "/api/runs.csv":
            return json.dumps([200, "text/csv; charset=utf-8", "﻿" + E.runs_csv(inv),
                               f"flame_in_freefall_runs{('_' + inv) if inv else ''}.csv"])
        if path == "/api/runs.json":
            data = {"note": "PSI experimental-table runs with provenance; see docs/SOURCE_REGISTER.md",
                    "runs": [r for r in E.runs if not inv or r["investigation"] == inv]}
            return json.dumps([200, "application/json; charset=utf-8", json.dumps(data, ensure_ascii=False, indent=1),
                               "flame_in_freefall_runs.json"])
        status, obj = _route(path, q)
    except Exception as e:  # mirror server.py: report errors as JSON
        status, obj = 500, {"error": f"{type(e).__name__}: {e}"}
    return json.dumps([status, "application/json; charset=utf-8", json.dumps(obj, ensure_ascii=False), None])
