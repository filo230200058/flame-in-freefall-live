"""Search, ranking, evidence and scenario logic for Flame in Freefall.

Runs fully locally on data/processed/*.json and the trained model artifact.
No network calls and no LLM. Every scientific statement returned carries a
citation (PSI record URL + field, or ZIP member + line range).
"""
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from textml import STOP, SoftmaxRegression, Tfidf  # noqa: E402

NR = "not_reported"
COMBUSTION = "Combustion Science"
MODEL_WEIGHT = 0.3            # fixed a priori; see docs/MODEL_CARD.md (not tuned on evaluation queries)
ABSTAIN_MIN_SCORE = 2.5       # BM25 score below which the assistant abstains
FIELD_WEIGHTS = {"title": 3.0, "acronym": 3.0, "keywords": 2.0, "objectives": 1.0, "approach": 1.0,
                 "hypothesis": 0.8, "research_impacts": 0.6, "hardware": 1.0, "flight_platform": 0.5,
                 "subresearch_area": 1.0}
QUESTION_STOP = set("""retrieve available information include including describe description explain tell show give find list
know want need please relevant related relationship analyze analyse identify collectively contribute contribution regarding
which what how why where who whom whose does did done shared gap gaps variable variable finding finding complementary overall
summary summarize summarise detail details file files""".split())
QUANT_RE = re.compile(r"\b(temperature|how (?:much|many|fast|long)|rate of|probability|value of|threshold|limit of|what percent)\b", re.I)
DISCLAIMER = ("Research discovery aid only. Relevance scores rank how well indexed PSI metadata matches your "
              "query; they are not fire risk, ignition probability, or evidence that any material is safe.")


def tok(text):
    out = []
    for t in re.findall(r"[a-z0-9][a-z0-9\-\.]*", (text or "").lower()):
        t = t.strip(".-")
        if not t or t in STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        """docs: list of {term: weighted tf}"""
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.len = [sum(d.values()) for d in docs]
        self.avg = sum(self.len) / max(self.N, 1)
        df = {}
        for d in docs:
            for t in d:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, i, qterms):
        d, s = self.docs[i], 0.0
        for t in qterms:
            tf = d.get(t)
            if tf:
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class Engine:
    def __init__(self, root=ROOT):
        P = root / "data" / "processed"
        self.records = json.loads((P / "records.json").read_text(encoding="utf-8"))
        self.chunks = json.loads((P / "chunks.json").read_text(encoding="utf-8"))
        self.by_id = {r["id"]: r for r in self.records}
        self.manifest = self._load(root / "data" / "raw" / "source_manifest.json")
        self.ingest_log = self._load(P / "ingest_log.json")
        self.metrics = self._load(root / "ml" / "artifacts" / "metrics.json")
        self.ranking_eval = self._load(root / "ml" / "artifacts" / "ranking_eval.json")
        rt = self._load(P / "runs.json") or {"tables": [], "runs": []}
        self.runs, self.run_tables = rt["runs"], rt["tables"]
        self.runs_by_rec = {}
        for u in self.runs:
            self.runs_by_rec.setdefault(u["investigation"], []).append(u)
        self.oof = {o["id"]: o for o in (self._load(root / "ml" / "artifacts" / "oof_predictions.json") or [])}
        art = self._load(root / "ml" / "artifacts" / "research_area_model.json")
        self.model_info = None
        if art:
            self.vec = Tfidf.from_dict(art["vectorizer"])
            self.model = SoftmaxRegression.from_dict(art["model"], len(self.vec.vocab))
            self.model_info = {k: art[k] for k in ("name", "version", "trained_utc", "dataset", "pretrained_components", "top_features")}
        # record-level weighted docs for BM25
        docs = []
        for r in self.records:
            d = {}
            for f, w in FIELD_WEIGHTS.items():
                text = " ".join(r["keywords"]) if f == "keywords" else self.val(r, f)
                for t in tok(text):
                    d[t] = d.get(t, 0) + w
            docs.append(d)
        self.bm25 = BM25(docs)
        cdocs = []
        for c in self.chunks:
            d = {}
            for t in tok(c["text"]):
                d[t] = d.get(t, 0) + 1
            cdocs.append(d)
        self.cbm25 = BM25(cdocs)
        self.chunks_by_record = {}
        for i, c in enumerate(self.chunks):
            self.chunks_by_record.setdefault(c["record_id"], []).append(i)

    @staticmethod
    def _load(p):
        p = pathlib.Path(p)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @staticmethod
    def val(r, f):
        v = r["fields"].get(f, {}).get("value", "")
        return "" if v in (NR, None) else v

    # ------------------------------------------------------------ model
    def classify_query(self, q):
        if not self.model_info or not q.strip():
            return None
        x = self.vec.transform_one(q)
        if not x:
            return {"known_terms": 0, "probs": None,
                    "note": "None of the query words are in the model vocabulary, so the model adds nothing."}
        p = self.model.predict_proba_one(x)
        inv = {i: t for t, i in self.vec.vocab.items()}
        return {"known_terms": len(x), "terms": sorted(inv[i] for i in x)[:20],
                "probs": dict(sorted(zip(self.model.classes, [round(v, 3) for v in p]), key=lambda kv: -kv[1]))}

    # ------------------------------------------------------------ helpers
    def card(self, r):
        d = r["derived"]
        return {"id": r["id"], "url": r["url"], "acronym": self.val(r, "acronym") or NR, "title": self.val(r, "title"),
                "research_area": self.val(r, "research_area") or NR, "subresearch_area": self.val(r, "subresearch_area") or NR,
                "project_type": self.val(r, "project_type") or NR, "platform": self.val(r, "flight_platform") or NR,
                "hardware": self.val(r, "hardware") or NR, "study_type": d["study_type"],
                "gravity_tested": d["gravity_tested"]["value"], "intent_tags": [t["tag"] for t in d["intent_tags"]],
                "materials": [m["material"] for m in d["materials"]], "has_local_archive": bool(r["local_files"]),
                "conditions": {k: v["status"] for k, v in d["conditions"].items()},
                "start_date": self.val(r, "start_date") or NR, "end_date": self.val(r, "end_date") or NR,
                "n_reports": len(r.get("literature", []))}

    def facets(self):
        def count(fn):
            out = {}
            for r in self.records:
                for v in fn(r):
                    out[v] = out.get(v, 0) + 1
            return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
        return {
            "research_area": count(lambda r: [self.val(r, "research_area") or NR]),
            "study_type": count(lambda r: [r["derived"]["study_type"]]),
            "gravity_tested": count(lambda r: [r["derived"]["gravity_tested"]["value"]]),
            "platform": count(lambda r: [self.val(r, "flight_platform") or NR]),
            "hardware": count(lambda r: [self.val(r, "hardware") or NR]),
            "materials": count(lambda r: [m["material"] for m in r["derived"]["materials"]]),
            "intent_tags": count(lambda r: [t["tag"] for t in r["derived"]["intent_tags"]]),
            "file_types": count(lambda r: [r["file_types"]["status"]]),
            "condition_mentioned": count(lambda r: [k for k, v in r["derived"]["conditions"].items() if v["status"] != "not_extracted"]),
        }

    def _filter(self, r, f):
        c = self.card(r)
        if f.get("area") and c["research_area"] != f["area"]:
            return False
        if f.get("study_type") and c["study_type"] != f["study_type"]:
            return False
        if f.get("gravity") and c["gravity_tested"] != f["gravity"]:
            return False
        if f.get("platform") and c["platform"] != f["platform"]:
            return False
        if f.get("hardware") and c["hardware"] != f["hardware"]:
            return False
        if f.get("material") and f["material"] not in c["materials"]:
            return False
        if f.get("tag") and f["tag"] not in c["intent_tags"]:
            return False
        if f.get("condition") and c["conditions"].get(f["condition"]) == "not_extracted":
            return False
        if f.get("file_type") and r["file_types"]["status"] != f["file_type"]:
            return False
        if f.get("psi") and f["psi"].lower() not in (r["id"] + " " + c["acronym"] + " " + c["title"]).lower():
            return False
        return True

    def evidence_for(self, rid, qterms, k=3):
        idx = self.chunks_by_record.get(rid, [])
        scored = sorted(((self.cbm25.score(i, qterms), i) for i in idx), reverse=True)
        out = []
        for s, i in scored[:k]:
            if s <= 0:
                break
            c = self.chunks[i]
            out.append({"text": c["text"], "field": c["field"], "location": c["location"], "url": c["url"],
                        "source_kind": c.get("source_kind"), "doc_title": c.get("doc_title"),
                        "score": round(s, 2), "matched_terms": sorted(set(qterms) & set(tok(c["text"])))})
        return out

    # ------------------------------------------------------------ search
    def search(self, q="", filters=None, limit=25, use_model=True):
        filters = filters or {}
        qterms = list(dict.fromkeys(tok(q)))
        qc = self.classify_query(q) if (use_model and q.strip()) else None
        probs = (qc or {}).get("probs") or {}
        results = []
        for i, r in enumerate(self.records):
            if not self._filter(r, filters):
                continue
            lex = self.bm25.score(i, qterms) if qterms else 0.0
            if qterms and lex <= 0:
                continue
            results.append([i, r, lex])
        mx = max((x[2] for x in results), default=0) or 1.0
        out = []
        for i, r, lex in results:
            area = self.val(r, "research_area")
            lexn = lex / mx if qterms else 0.0
            pm = probs.get(area, 0.0) if probs else 0.0
            score = (1 - MODEL_WEIGHT) * lexn + MODEL_WEIGHT * pm if (qterms and probs) else lexn
            d = self.bm25.docs[i]
            matched = [t for t in qterms if t in d]
            missing_q = [t for t in qterms if t not in d]
            c = self.card(r)
            c.update({
                "score": round(score, 4),
                "explanation": {
                    "lexical_bm25": round(lex, 3), "lexical_normalized": round(lexn, 3),
                    "model_p_area_given_query": round(pm, 3) if probs else None,
                    "model_weight": MODEL_WEIGHT if probs else 0.0,
                    "matched_terms": matched, "query_terms_not_found": missing_q,
                    "structured_filters_applied": {k: v for k, v in filters.items() if v},
                },
                "evidence": self.evidence_for(r["id"], qterms) if qterms else [],
                "missing_dimensions": [k for k, v in c["conditions"].items() if v == "not_extracted" and
                                       (r["derived"].get("publication_conditions") or {}).get(k, {}).get("status", "not_extracted") == "not_extracted"]
                                      + r["missing_fields"],
            })
            out.append(c)
        out.sort(key=lambda c: (-c["score"], int(c["id"].split("-")[1])))
        return {"query": q, "query_terms": qterms, "filters": filters, "n_results": len(out),
                "query_classification": qc, "results": out[:limit], "disclaimer": DISCLAIMER,
                "ranking_formula": (f"score = {1-MODEL_WEIGHT:.1f} x normalized BM25 + {MODEL_WEIGHT:.1f} x model P(record's official research area | query)"
                                    if probs else "score = normalized BM25 (model not applied)")}

    # ------------------------------------------------------------ detail / compare
    def detail(self, rid):
        r = self.by_id.get(rid)
        if not r:
            return None
        out = dict(r)
        out["card"] = self.card(r)
        out["sensors"] = self.sensors_for(rid)
        out["variables"] = self.run_variables(rid)
        out["model_check"] = self.oof.get(rid)
        return out

    COMPARE_FIELDS = [("acronym", "Acronym"), ("research_area", "Research area"), ("subresearch_area", "Sub-area"),
                      ("project_type", "Project type"), ("flight_platform", "Flight platform"), ("hardware", "Hardware"),
                      ("start_date", "Start"), ("end_date", "End"), ("managing_center", "Managing center")]

    def compare(self, ids):
        recs = [self.by_id[i] for i in ids if i in self.by_id]
        rows = []
        for f, label in self.COMPARE_FIELDS:
            rows.append({"label": label, "cells": [self._cell(r, f) for r in recs]})
        rows.append({"label": "Study type (derived)", "cells": [{"value": r["derived"]["study_type"], "derived": True} for r in recs]})
        rows.append({"label": "Gravity regime tested (derived)", "cells": [{"value": r["derived"]["gravity_tested"]["value"], "derived": True,
                                                                        "basis": r["derived"]["gravity_tested"]["basis"]} for r in recs]})
        for cond in ("oxygen", "pressure", "airflow"):
            cells = []
            for r in recs:
                cd = r["derived"]["conditions"][cond]
                if cd["status"] == "not_extracted":
                    cells.append({"value": "Not reported", "missing": True})
                else:
                    vals = [v["text"] for m in cd["mentions"] for v in m["values"]]
                    m0 = cd["mentions"][0]
                    cells.append({"value": (", ".join(dict.fromkeys(vals)) if vals else "mentioned, no value given"),
                                  "quote": m0["sentence"], "location": m0["location"], "url": r["url"], "derived": True})
            rows.append({"label": f"{cond.title()} (narrative mention)", "cells": cells})
        rows.append({"label": "Materials mentioned", "cells": [
            {"value": ", ".join(m["material"] for m in r["derived"]["materials"]) or "Not reported",
             "missing": not r["derived"]["materials"], "derived": True} for r in recs]})
        rows.append({"label": "Sensors / measurement types", "cells": [
            {"value": ", ".join(s["sensor"] for s in self.sensors_for(r["id"])) or "Not reported",
             "missing": not self.sensors_for(r["id"]), "derived": True} for r in recs]})
        rows.append({"label": "Measured test runs indexed", "cells": [
            {"value": str(len(self.runs_by_rec.get(r["id"], []))) if self.runs_by_rec.get(r["id"]) else "None indexed",
             "missing": not self.runs_by_rec.get(r["id"]), "derived": True} for r in recs]})
        cells = []
        for r in recs:
            mp = r["derived"].get("measured_pressure")
            cells.append({"value": f"{mp['kpa_mean_range'][0]}-{mp['kpa_mean_range'][1]} kPa (test {mp['test_id']}, {mp['date']})",
                          "location": mp["location"]} if mp else {"value": "Not reported", "missing": True})
        rows.append({"label": "Measured chamber pressure (PSI data file)", "cells": cells})
        rows.append({"label": "Data files inspected", "cells": [{"value": r["file_types"].get("note", ""), "derived": True} for r in recs]})
        return {"records": [self.card(r) for r in recs], "rows": rows,
                "note": "Cells show source values only. 'Not reported' means the indexed metadata does not state it; nothing is inferred."}

    def _cell(self, r, f):
        fd = r["fields"].get(f)
        if not fd or fd["status"] == NR:
            return {"value": "Not reported", "missing": True}
        c = {"value": fd["value"], "location": fd["source"]["location"], "url": fd["source"].get("url", r["url"])}
        if fd["status"] == "conflicting":
            c["conflict"] = fd.get("conflict")
        return c

    # ------------------------------------------------------------ assistant
    # ---- entity detection: which investigations does the question name?
    def detect_entities(self, q):
        found = []
        for m in re.finditer(r"\bPSI[- ]?(\d+)\b", q, re.I):
            rid = f"PSI-{m.group(1)}"
            if rid in self.by_id:
                found.append(rid)
        for r in self.records:
            acr = self.val(r, "acronym")
            if not acr:
                continue
            for a in [x.strip() for x in re.split(r"[/;]", acr) if x.strip()]:
                flags = 0 if (a.upper() in ("SAME", "FLAME", "FIRE") or len(a) <= 3) else re.I
                if re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", q, flags):
                    found.append(r["id"])
                    break
        # test IDs from parsed PSI data files (e.g. 'FLEX-263', 'C11H201') -> their investigation
        for r in self.records:
            for f in r.get("data_files", []):
                tid = f.get("test_id")
                if tid and tid != "unknown" and re.search(r"(?<![\w-])" + re.escape(tid) + r"(?![\w-])", q, re.I):
                    found.append(r["id"])
        # family names with no exact acronym (e.g. "SAFFIRE" -> SAFFIRE-I, -II, -III)
        exact = {self.val(r, "acronym").upper() for r in self.records if self.val(r, "acronym")}
        fams = {}
        for r in self.records:
            acr = self.val(r, "acronym")
            if acr:
                fams.setdefault(re.split(r"[\s/;\-]", acr)[0].upper(), []).append(r["id"])
        for base, members in fams.items():
            if len(members) > 1 and base not in exact and len(base) > 3 and \
                    re.search(r"(?<![\w-])" + re.escape(base) + r"(?![\w-])", q, re.I):
                found.extend(members)
        return list(dict.fromkeys(found))

    def entity_terms(self, ids):
        out = set()
        for i in ids:
            for a in re.split(r"[\s/;\-]", self.val(self.by_id[i], "acronym") or ""):
                out.update(tok(a))
            out.update(tok(i))
        return out

    def _fact(self, c, rec_id):
        r = self.by_id[rec_id]
        return {"statement": c["text"], "record_id": rec_id, "acronym": self.val(r, "acronym") or NR,
                "field": c["field"], "location": c["location"], "url": c["url"],
                "source_kind": c.get("source_kind"), "doc_title": c.get("doc_title"), "link_level": c.get("link_level"),
                "gravity_tested": r["derived"]["gravity_tested"]["value"]}

    def _field_fact(self, r, f, first_sentence_only=False):
        fd = r["fields"].get(f)
        if not fd or fd["status"] == NR:
            return None
        text = fd["value"]
        if first_sentence_only:
            ss = [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if len(s) > 30]
            text = ss[0] if ss else text
        return {"statement": text, "record_id": r["id"], "acronym": self.val(r, "acronym") or NR, "field": f,
                "location": fd["source"]["location"], "url": fd["source"].get("url", r["url"]), "source_kind": "psi_metadata",
                "gravity_tested": r["derived"]["gravity_tested"]["value"]}

    def profile(self, rid):
        """Everything indexed about one investigation, each item cited. No generated text."""
        r = self.by_id[rid]
        sections = []
        for f, label in (("objectives", "Scientific objectives"), ("approach", "Experiment description (approach)"),
                         ("hypothesis", "Hypothesis"), ("research_impacts", "Research impacts")):
            ff = self._field_fact(r, f)
            sections.append({"label": label, "facts": [ff] if ff else [], "missing": ff is None})
        attrs = []
        for f, label in (("project_type", "Project type"), ("flight_platform", "Flight platform"), ("hardware", "Hardware"),
                         ("start_date", "Start"), ("end_date", "End"), ("managing_center", "Managing center")):
            fd = r["fields"].get(f)
            attrs.append({"label": label, "value": fd["value"] if fd and fd["status"] != NR else "Not reported",
                          "location": fd["source"]["location"] if fd else None, "missing": not fd or fd["status"] == NR})
        files = []
        for lf in r["local_files"]:
            files.append({"kind": "local archive (metadata only)", "name": lf["file"], "detail": "; ".join(lf["members"])})
        files.append({"kind": "PSI data files", "name": r["url"], "detail": r["file_types"].get("note", "")})
        for l in r.get("literature", []):
            files.append({"kind": "public NASA report", "name": f"NTRS {l['ntrs_id']}: {l['title']}", "url": l["url"],
                          "detail": f"{l['link_level']}; indexed: {l['text_indexed']}"})
        pubs = [{"title": p.get("title"), "doi": p.get("doi")} for p in r.get("publications", [])][:12]
        return {"id": rid, "acronym": self.val(r, "acronym") or NR, "title": self.val(r, "title"), "url": r["url"],
                "gravity_tested": r["derived"]["gravity_tested"]["value"], "sections": sections, "attributes": attrs,
                "files": files, "publications": pubs, "missing_fields": r["missing_fields"]}

    def compare_entities(self, ids):
        """Shared variables and gaps across named investigations, computed from indexed data only."""
        recs = [self.by_id[i] for i in ids]
        dims = []
        for k in ("oxygen", "pressure", "airflow"):
            row = {}
            for r in recs:
                md = r["derived"]["conditions"][k]
                pd = (r["derived"].get("publication_conditions") or {}).get(k, {})
                if md["has_numeric_value"] or pd.get("has_numeric_value"):
                    src = md if md["has_numeric_value"] else pd
                    m = next(m for m in src["mentions"] if m["values"])
                    source = "PSI metadata"
                    if src is not md:
                        nid = re.search(r"NTRS (\d+)", m["location"])
                        lvl = next((l["link_level"] for l in r.get("literature", []) if nid and l["ntrs_id"] == nid.group(1)), "")
                        source = ("linked NASA report (about a related investigation in the same family, not this one directly)"
                                  if lvl.startswith("family") else "linked NASA report")
                    row[r["id"]] = {"status": "value stated", "value": ", ".join(v["text"] for v in m["values"]),
                                    "quote": m["sentence"], "location": m["location"], "source": source}
                elif md["status"] != "not_extracted" or pd.get("status", "not_extracted") != "not_extracted":
                    row[r["id"]] = {"status": "mentioned, no value"}
                else:
                    row[r["id"]] = {"status": "not reported"}
            dims.append({"dimension": k, "cells": row})
        dims.append({"dimension": "gravity regime tested", "cells": {r["id"]: {"status": r["derived"]["gravity_tested"]["value"]} for r in recs}})
        def mats(r):
            meta = [m["material"] for m in r["derived"]["materials"]]
            pub = [m["material"] for m in r["derived"].get("publication_materials", []) if m["material"] not in meta]
            s = ", ".join(meta)
            if pub:
                s += ("; " if s else "") + "in linked reports: " + ", ".join(pub[:8])
            return s or "not reported"
        dims.append({"dimension": "fuel / material", "cells": {r["id"]: {"status": mats(r)} for r in recs}})
        dims.append({"dimension": "research intent (reviewed tags)", "cells": {r["id"]: {"status": ", ".join(
            t["tag"].replace("_", " ") + (" (context)" if (t.get("review") or {}).get("verdict") == "partial" else "")
            for t in r["derived"]["intent_tags"]) or "none"} for r in recs}})
        shared = [d["dimension"] for d in dims[:3] if all(c["status"] == "value stated" for c in d["cells"].values())]
        gaps = [d["dimension"] for d in dims[:3] if all(c["status"] == "not reported" for c in d["cells"].values())]
        regimes = sorted(set(r["derived"]["gravity_tested"]["value"] for r in recs))
        tags = {}
        for r in recs:
            for t in r["derived"]["intent_tags"]:
                if (t.get("review") or {}).get("verdict") != "partial":
                    tags.setdefault(t["tag"].replace("_", " "), []).append(self.val(r, "acronym") or r["id"])
        roles = [f"{k}: {', '.join(v)}" for k, v in sorted(tags.items())]
        notes = [
            "Roles below come from the reviewed research-intent tags, not from a reading of results.",
            "All named investigations share the gravity regime: " + regimes[0] + "." if len(regimes) == 1
            else "The investigations ran in different gravity regimes (" + "; ".join(regimes) + "); they are not interchangeable.",
        ]
        if shared:
            notes.append("Variables with a stated value for every investigation: " + ", ".join(shared) + ".")
        if gaps:
            notes.append("Not reported for any of them in indexed sources: " + ", ".join(gaps) + ".")
        notes.append("No indexed investigation was conducted in lunar or Martian gravity, so extrapolation to those regimes is unsupported.")
        return {"records": [{"id": r["id"], "acronym": self.val(r, "acronym") or NR, "title": self.val(r, "title"),
                             "objective": self._field_fact(r, "objectives", first_sentence_only=True)} for r in recs],
                "dimensions": dims, "roles": roles, "notes": notes}

    PROFILE_RE = re.compile(r"\b(information|overview|about|describe|description|objective|summar|tell me|what is|what was|files?|details?)\b", re.I)

    def ask(self, q, max_facts=6):
        ents = self.detect_entities(q)
        if len(ents) >= 2:
            cmpd = self.compare_entities(ents[:6])
            names = [r["acronym"] if r["acronym"] != NR else r["id"] for r in cmpd["records"]]
            runs_n = {self.val(self.by_id[i], "acronym") or i: len(self.runs_by_rec.get(i, [])) for i in ents[:6]}
            with_runs = [k for k, v in runs_n.items() if v]
            concl = {"level": "partial",
                     "headline": f"{len(names)} investigations compared: {', '.join(names)}.",
                     "points": cmpd["roles"][:4] + cmpd["notes"][1:] +
                               ([f"Measured test runs exist for: {', '.join(f'{k} ({runs_n[k]})' for k in with_runs)}; the others rely on descriptions and reports."]
                                if with_runs else ["None of these has an indexed experimental table; values come from descriptions and reports."]),
                     "method": "Computed from the comparison table with fixed rules; no text is generated."}
            return {"status": "comparison", "mode": "comparison", "entities": ents, "comparison": cmpd, "conclusion": concl,
                    "evidence": self._ask_retrieval(q, max_facts=6, restrict=ents),
                    "method": "Named investigations were recognised; the table is computed from their indexed records. "
                              "Quoted sentences are retrieved from those records only. No text is generated.",
                    "disclaimer": DISCLAIMER}
        names_test = any(re.search(r"(?<![\w-])" + re.escape(f["test_id"]) + r"(?![\w-])", q, re.I)
                         for rid in ents for f in self.by_id[rid].get("data_files", []) if f.get("test_id") not in (None, "unknown"))
        if len(ents) == 1 and names_test:
            return self._ask_retrieval(q, max_facts, restrict=ents)
        if len(ents) == 1 and self.PROFILE_RE.search(q):
            prof = self.profile(ents[0])
            r = self.by_id[ents[0]]
            rr = self.runs_by_rec.get(ents[0], [])
            reported = sum(1 for s_ in prof["sections"] if not s_["missing"])
            pts = [f"{prof['acronym'] if prof['acronym'] != NR else prof['id']} is a {r['derived']['study_type']} conducted as {prof['gravity_tested']}.",
                   f"{reported} of {len(prof['sections'])} description fields are reported; {len(r.get('literature', []))} public NASA report(s) and "
                   f"{len(r.get('data_files', []))} PSI data file(s) are linked."]
            if rr:
                oc = [u for u in rr if (u.get("outcome") or {}).get("status") == "reported"]
                pts.append(f"{len(rr)} measured test runs are indexed" + (f"; {sum(1 for u in oc if u['outcome']['ignition'] == 'sustained')} of {len(oc)} "
                                                                         f"reported outcomes had a sustained flame" if oc else "") + " (see the Test runs tab).")
            else:
                pts.append("No experimental table is indexed for it, so no run-level measurements are available here.")
            if r["missing_fields"]:
                pts.append("Not reported in the record: " + ", ".join(f.replace("_", " ") for f in r["missing_fields"]) + ".")
            concl = {"level": "supported", "headline": f"Everything indexed about {prof['id']} is below, each part with its source.",
                     "points": pts, "method": "Computed from the record with fixed rules; no text is generated."}
            return {"status": "profile", "mode": "profile", "entities": ents, "profile": prof, "conclusion": concl,
                    "method": "The question names one investigation, so its indexed record is shown field by field, each with its source.",
                    "disclaimer": DISCLAIMER}
        return self._ask_retrieval(q, max_facts, restrict=ents or None)

    def _ask_retrieval(self, q, max_facts=6, restrict=None):
        drop = QUESTION_STOP | (self.entity_terms(restrict) if restrict else set())
        qterms = [t for t in dict.fromkeys(tok(q)) if t not in drop]
        if not qterms:
            return {"status": "abstain", "reason": "The question has no searchable terms.", "facts": []}
        test_ids = [f["test_id"] for rid in (restrict or []) for f in self.by_id[rid].get("data_files", [])
                    if f.get("test_id") and re.search(r"(?<![\w-])" + re.escape(f["test_id"]) + r"(?![\w-])", q, re.I)]
        wants_number = bool(QUANT_RE.search(q) or re.search(r"concentration|velocit|speed|pressure|how (?:much|many)|level|value", q, re.I))
        if restrict:
            sr = {"results": [dict(self.card(self.by_id[i]), score=1.0) for i in restrict]}
        else:
            sr = self.search(" ".join(qterms), limit=8)
        cand = []
        for res in sr["results"]:
            for i in self.chunks_by_record.get(res["id"], []):
                s = self.cbm25.score(i, qterms)
                if s > 0:
                    c = self.chunks[i]
                    fam = 0.5 if (c.get("link_level") or "").startswith("family") else 1.0
                    if c.get("source_kind") == "psi_data_file" and any(tid in c["text"] for tid in test_ids):
                        s += 50.0          # the question names this exact test: its own data row comes first
                    num = 1.6 if wants_number and re.search(r"\d+(?:\.\d+)?\s*(?:%|kPa|psi|atm|cm/s|mm/s|m/s|K\b|°C|s\b|mm\b)", c["text"]) else 1.0
                    cand.append((s * fam * num + 2.0 * res["score"], s, i, res))
        cand.sort(key=lambda x: -x[0])
        facts, seen = [], set()
        need = min(2, len(qterms))
        for _, s, i, res in cand:
            c = self.chunks[i]
            if s < ABSTAIN_MIN_SCORE or c["text"] in seen:
                continue
            if len(set(qterms) & set(tok(c["text"]))) < need and not (c.get("source_kind") == "psi_data_file" and s >= 50):
                continue
            seen.add(c["text"])
            f = self._fact(c, c["record_id"])
            f.update({"score": round(s, 2), "matched_terms": sorted(set(qterms) & set(tok(c["text"])))})
            facts.append(f)
            if len(facts) >= max_facts:
                break
        covered = set(t for f in facts for t in f["matched_terms"])
        uncovered = [t for t in qterms if t not in covered]
        if not facts:
            return {"status": "abstain", "facts": [], "query_terms": qterms,
                    "reason": "No indexed PSI metadata sentence matches this question strongly enough. The catalog may not cover it."}
        regimes = sorted(set(f["gravity_tested"] for f in facts))
        kinds = set(f["source_kind"] for f in facts)
        interp = []
        if kinds - {"ntrs_report"}:
            interp.append("Sentences marked PSI metadata describe what each investigation set out to study, not measured results.")
        if "ntrs_report" in kinds:
            interp.append("Sentences marked NASA report are quoted from public NTRS reports linked to the investigation. "
                          "They may state results; read the cited page for test conditions, units and uncertainty before relying on a number.")
        if len(regimes) > 1:
            interp.append("The cited records were run in different gravity regimes (" + "; ".join(regimes) + "), so they are not interchangeable.")
        if re.search(r"\blunar|\bmoon\b|\bmars\b|martian|partial gravity", q, re.I):
            interp.append("No indexed investigation was conducted in lunar, Martian or other partial gravity. Microgravity and "
                          "normal-gravity results above are not a substitute for those regimes.")
        if uncovered:
            interp.append("No cited sentence addresses these query terms: " + ", ".join(uncovered) + ".")
        if QUANT_RE.search(q):
            interp.append("The question asks for a quantity. This assistant only quotes sentences; it does not compute or "
                          "infer a value. Any number above is as stated on the cited page, under that study's conditions.")
        if len(covered) < len(qterms) / 2:
            return {"status": "abstain", "facts": [], "query_terms": qterms, "uncovered_terms": uncovered,
                    "reason": "Indexed evidence covers fewer than half of the question's terms (" +
                              ", ".join(sorted(covered)) + " found; " + ", ".join(uncovered) + " not found). "
                              "Answering would mean guessing."}
        status = "answered" if not uncovered and not QUANT_RE.search(q) else "partial"
        out = {"status": status, "facts": facts, "interpretation": interp, "query_terms": qterms,
               "uncovered_terms": uncovered, "method": "Extractive: BM25 sentence retrieval within top-ranked records. No text is generated.",
               "disclaimer": DISCLAIMER}
        out["conclusion"] = self._answer_conclusion(out)
        return out

    def _answer_conclusion(self, a):
        """Template conclusion for an extractive answer: what the quoted evidence covers, where it is from, what is missing."""
        facts = a["facts"]
        recs = []
        for f in facts:
            if f["record_id"] not in recs:
                recs.append(f["record_id"])
        names = [(self.val(self.by_id[r], "acronym") or r) for r in recs]
        kinds = {}
        for f in facts:
            k = {"psi_api": "PSI metadata", "local_zip": "PSI metadata", "ntrs_report": "NASA reports",
                 "psi_data_file": "PSI measured data", "psi_document": "PSI planning documents"}.get(f.get("source_kind"), "PSI metadata")
            kinds[k] = kinds.get(k, 0) + 1
        top = max(recs, key=lambda r: sum(1 for f in facts if f["record_id"] == r)) if recs else None
        regimes = sorted(set(f["gravity_tested"] for f in facts))
        pts = [f"The {len(facts)} quoted facts come from {len(recs)} investigation(s): {', '.join(names)}; "
               f"most come from {self.val(self.by_id[top], 'acronym') or top}." if top else "No facts."]
        pts.append("Sources: " + ", ".join(f"{v} from {k}" for k, v in kinds.items()) + ".")
        if any(k in kinds for k in ("NASA reports", "PSI measured data")):
            pts.append("Some facts state measured results; check the cited page or file for that test’s conditions before reusing a number.")
        else:
            pts.append("All facts are investigation descriptions (what was studied), not measured results.")
        pts.append("Gravity regime of the cited work: " + "; ".join(regimes) + ".")
        if a["uncovered_terms"]:
            pts.append("Parts of the question with no supporting sentence: " + ", ".join(a["uncovered_terms"]) + ".")
        head = ("The indexed evidence addresses the question; read the quoted facts below."
                if a["status"] == "answered" else
                "The indexed evidence answers the question only in part; the gaps are listed below.")
        return {"level": "supported" if a["status"] == "answered" else "partial", "headline": head, "points": pts,
                "method": "Computed from the quoted facts with fixed rules; no text is generated."}

    # ------------------------------------------------------------ scenario
    def scenario(self, gravity="", oxygen=None, pressure_kpa=None, airflow_cms=None, material="", focus="", scope="combustion"):
        pool = [r for r in self.records if scope != "combustion" or self.val(r, "research_area") == COMBUSTION]
        dims_requested = [d for d, v in (("gravity", gravity), ("oxygen", oxygen), ("pressure", pressure_kpa),
                                          ("airflow", airflow_cms), ("material", material), ("focus", focus)) if v not in (None, "")]
        rows = []
        for r in pool:
            d = r["derived"]
            dims = {}
            if gravity:
                dims["gravity"] = self._gravity_match(r, gravity)
            for key, req, unit_conv in (("oxygen", oxygen, None), ("pressure", pressure_kpa, "kpa"), ("airflow", airflow_cms, "cms")):
                if req is None:
                    continue
                dims[key] = self._cond_match(r, key, req, unit_conv)
            if material:
                rhit = [u for u in self.runs_by_rec.get(r["id"], []) if material.lower() in (u.get("material_text") or "").lower()]
                hit = [m for m in d["materials"] if material.lower() in m["material"]]
                if rhit:
                    dims["material"] = {"status": "match", "detail": f"{len(rhit)} test run(s) in the experimental table used it, e.g. {rhit[0]['run_id']}: {rhit[0]['material_text']}",
                                        "location": f"{rhit[0]['source']['file']}, line {rhit[0]['source']['line']}"}
                    hit = None
                phit = [m for m in d.get("publication_materials", []) if material.lower() in m["material"]]
                if hit is None:
                    pass
                elif hit:
                    dims["material"] = {"status": "match", "detail": hit[0]["sentence"], "location": hit[0]["location"]}
                elif phit:
                    dims["material"] = {"status": "match", "detail": "named in a linked NASA report: " + phit[0]["sentence"],
                                        "location": phit[0]["location"]}
                else:
                    dims["material"] = {"status": "not_reported", "detail": f"'{material}' not mentioned in metadata or linked reports"}
            if focus:
                qt = tok(focus)
                ev = self.evidence_for(r["id"], qt, k=1)
                dims["focus"] = ({"status": "match", "detail": ev[0]["text"], "location": ev[0]["location"]} if ev
                                 else {"status": "not_reported", "detail": "focus terms not found"})
            n_match = sum(1 for v in dims.values() if v["status"] == "match")
            n_partial = sum(1 for v in dims.values() if v["status"] == "partial")
            rows.append({"record": self.card(r), "dimensions": dims, "n_match": n_match, "n_partial": n_partial})
        rows.sort(key=lambda x: (-x["n_match"], -x["n_partial"], int(x["record"]["id"].split("-")[1])))
        unsupported = [dname for dname in dims_requested if not any(x["dimensions"].get(dname, {}).get("status") == "match" for x in rows)]
        lim = ["Oxygen, air flow and material are matched first against measured PSI experimental tables (BASS-II, SAFFIRE-II, SAME; file and line cited). "
               "Other investigations fall back to metadata narrative or linked NASA report sentences, which are text as written, not run tables.",
               "Microgravity, lunar, Martian and normal gravity are distinct regimes. Evidence from one is not a substitute for another."]
        if gravity in ("lunar", "martian"):
            n_any = sum(1 for x in rows if x["dimensions"].get("gravity", {}).get("status") == "match")
            lim.append(f"{n_any} of {len(rows)} records in scope report being conducted in {gravity} gravity.")
        out = {"scope": f"{len(pool)} {'Combustion Science' if scope == 'combustion' else 'PSI'} investigations indexed",
                "requested": {"gravity": gravity or None, "oxygen_percent": oxygen, "pressure_kpa": pressure_kpa,
                              "airflow_cm_s": airflow_cms, "material": material or None, "focus": focus or None},
                "rows": rows, "unsupported_dimensions": unsupported, "limitations": lim,
                "note": "This is an evidence explorer, not a fire-risk prediction. Match counts show how much indexed evidence relates to each part of your scenario."}
        out["conclusion"] = self._scenario_conclusion(out, dims_requested, gravity, oxygen)
        return out

    def _scenario_conclusion(self, s, dims, gravity, oxygen):
        """Template conclusion computed from the rows above. No free-text generation."""
        rows = s["rows"]
        n = len(rows)
        per = []
        for d in dims:
            m = sum(1 for r in rows if r["dimensions"].get(d, {}).get("status") == "match")
            p = sum(1 for r in rows if r["dimensions"].get(d, {}).get("status") == "partial")
            per.append({"dimension": d, "match": m, "partial": p,
                        "verdict": "supported" if m else ("partial evidence only" if p else "no evidence")})
        sup = [x["dimension"] for x in per if x["verdict"] == "supported"]
        part = [x["dimension"] for x in per if x["verdict"] == "partial evidence only"]
        none = [x["dimension"] for x in per if x["verdict"] == "no evidence"]
        core_missing = [d for d in dims if d in ("gravity", "oxygen", "pressure", "airflow") and d not in sup]
        if dims and len(sup) == len(dims):
            overall, level = "Every part of this scenario has direct indexed evidence.", "supported"
        elif core_missing:
            overall, level = (f"Not supported for the conditions that define this environment ({', '.join(core_missing)}). "
                              + (f"Related evidence exists only for {', '.join(sup)}." if sup else "")), "unsupported"
        elif sup:
            overall, level = (f"Partly supported: direct evidence exists for {', '.join(sup)}, but not for "
                              f"{', '.join(part + none)}."), "partial"
        else:
            overall, level = "Not supported: no indexed record gives direct evidence for this scenario.", "unsupported"
        points = []
        if gravity and gravity != "microgravity" and "gravity" not in sup:
            points.append(f"No indexed investigation was conducted in {gravity} gravity; microgravity results are not a substitute.")
        if oxygen is not None:
            o2 = [u["o2_initial"]["value"] for u in self.runs if u.get("o2_initial", {}).get("status") == "reported"]
            if o2:
                near = sum(1 for v in o2 if abs(v - oxygen) <= 3)
                points.append(f"Measured test runs span {min(o2)}–{max(o2)}% O2; {near} of {len(o2)} runs are within ±3 points of {oxygen}%."
                              + (" The requested level is outside the tested range." if oxygen > max(o2) + 3 or oxygen < min(o2) - 3 else ""))
        if oxygen is not None and oxygen >= 25:
            elev = [r for r in rows if r["dimensions"].get("oxygen", {}).get("elevated_o2")]
            if elev:
                points.append("Elevated-oxygen tests are described for: " + ", ".join(
                    (r["record"]["acronym"] if r["record"]["acronym"] != NR else r["record"]["id"]) for r in elev)
                    + ". These are the closest high-O2 evidence; check fuel type (often gaseous flames, not solid materials) on the cited pages.")
        if gravity in ("lunar", "martian"):
            both = [r for r in self.records if self.val(r, "research_area") == COMBUSTION
                    and r["derived"]["gravity_tested"]["value"].startswith("microgravity")
                    and (r["derived"]["gravity_mentions"].get("normal_gravity") or (r["derived"].get("publication_gravity_mentions") or {}).get("normal_gravity"))]
            if both:
                points.append(f"No partial-gravity data exists in PSI. {len(both)} microgravity investigations also discuss normal-gravity behaviour ("
                              + ", ".join((self.val(r, 'acronym') or r['id']) for r in both[:8])
                              + "); together they can help bracket, not predict, behaviour at intermediate gravity.")
        best = [r for r in rows if r["n_match"] or r["n_partial"]][:3]
        if best:
            points.append("Closest evidence: " + "; ".join(
                f"{r['record']['acronym'] if r['record']['acronym'] != NR else r['record']['id']} ({r['n_match']} match, {r['n_partial']} partial)"
                for r in best) + ".")
        points.append("Use this to find what was measured, not to judge whether a design is safe.")
        return {"level": level, "headline": overall, "per_dimension": per, "points": points,
                "method": "Computed from the matrix above with fixed rules; no text is generated."}

    def _gravity_match(self, r, want):
        tested = r["derived"]["gravity_tested"]["value"]
        ment = r["derived"]["gravity_mentions"]
        key = {"microgravity": "microgravity", "lunar": "lunar_gravity", "martian": "martian_gravity", "normal": "normal_gravity"}.get(want)
        if want == "microgravity" and tested.startswith("microgravity"):
            return {"status": "match", "detail": tested, "basis": r["derived"]["gravity_tested"]["basis"]}
        pment = r["derived"].get("publication_gravity_mentions") or {}
        for src, mm in (("metadata narrative", ment), ("a linked NASA report", pment)):
            if key and mm.get(key):
                m = mm[key][0]
                return {"status": "partial", "detail": f"{want} gravity is mentioned in {src}, but the investigation was conducted as: {tested}",
                        "quote": m["sentence"], "location": m["location"]}
        return {"status": "different_regime" if tested != "unknown" else "unknown",
                "detail": f"Conducted as: {tested}. Not evidence for {want} gravity."}

    ELEV_O2 = re.compile(r"elevated oxygen|oxygen[- ]enriched|enriched oxygen|increased oxygen|"
                         r"oxygen concentrations? (?:of|up to|as high as) (?<![\d.])(?:[3-9]\d|100)\s*%|(?<![\d.])(?:[3-9]\d|100)\s*%\s*(?:oxygen|O2)", re.I)
    CITATION_LIKE = re.compile(r"ASTM|\bSTP\s?\d|\bpp\.|\bVol\.|et al\.|Proc\.|Journal|J\. [A-Z]", re.I)

    def _cond_match(self, r, key, req, conv):
        res = self._cond_match_core(r, key, req, conv)
        if key == "oxygen" and req is not None and req >= 25 and res.get("status") != "match":
            for i in self.chunks_by_record.get(r["id"], []):
                c = self.chunks[i]
                if c.get("source_kind") in ("psi_document",) or (c.get("link_level") or "").startswith("family"):
                    continue
                if self.ELEV_O2.search(c["text"]) and not self.CITATION_LIKE.search(c["text"]):
                    return {"status": "partial", "source": "elevated-oxygen test described",
                            "detail": "describes tests at elevated oxygen (above air); the exact level and how it compares to the request must be read on the cited page",
                            "quote": c["text"][:300], "location": c["location"], "elevated_o2": True}
        return res

    def _cond_match_core(self, r, key, req, conv):
        mp = r["derived"].get("measured_pressure")
        if key == "pressure" and mp:
            lo, hi = mp["kpa_mean_range"]
            st_ = "match" if lo - 15 <= req <= hi + 15 else "partial"
            return {"status": st_, "source": "measured PSI data file",
                    "detail": f"measured chamber pressure {lo}-{hi} kPa (from {mp['psia_mean_range'][0]}-{mp['psia_mean_range'][1]} psia) in test {mp['test_id']}"
                              + ("" if st_ == "match" else f" (differs from requested {req})"),
                    "quote": mp["note"], "location": mp["location"]}
        tr = [u for u in self.runs_by_rec.get(r["id"], []) if u.get("o2_initial", {}).get("status") == "reported"]
        if key == "oxygen" and tr:
            best = min(tr, key=lambda u: abs(u["o2_initial"]["value"] - req))
            v = best["o2_initial"]["value"]
            n_near = sum(1 for u in tr if abs(u["o2_initial"]["value"] - req) <= 3)
            lo, hi = min(u["o2_initial"]["value"] for u in tr), max(u["o2_initial"]["value"] for u in tr)
            st_ = "match" if n_near else "partial"
            return {"status": st_, "source": "PSI experimental table (measured)",
                    "detail": (f"{n_near} of {len(tr)} test runs had initial O2 within ±3 points of {req}%; nearest: test {best['run_id']} at {v}%"
                               if n_near else f"no test run near {req}%: the {len(tr)} runs span {lo}-{hi}% O2 (nearest {best['run_id']} at {v}%)"),
                    "quote": f"test {best['run_id']}: {best['material_text']}, initial O2 {best['o2_initial']['raw']}",
                    "location": f"{best['source']['file']}, line {best['source']['line']}"}
        if key == "airflow" and self.runs_by_rec.get(r["id"]):
            fr = [u for u in self.runs_by_rec[r["id"]] if u.get("airflow", {}).get("status") == "reported"]
            if fr:
                best = min(fr, key=lambda u: abs(u["airflow"]["value"] - req))
                st_ = "match" if abs(best["airflow"]["value"] - req) <= 5 else "partial"
                return {"status": st_, "source": "PSI experimental table (measured)",
                        "detail": f"test {best['run_id']} air flow {best['airflow']['raw']} cm/s",
                        "quote": f"test {best['run_id']}: {best['material_text']}", "location": f"{best['source']['file']}, line {best['source']['line']}"}
        runs = r["derived"].get("runs") or []
        if key == "oxygen" and runs:
            best = min(runs, key=lambda u: abs(u["Y_O2"] * 100 - req) if isinstance(u["Y_O2"], float) else 1e9)
            if isinstance(best["Y_O2"], float):
                close = abs(best["Y_O2"] * 100 - req) <= 3
                return {"status": "partial", "source": "PSI test summary sheet",
                        "detail": f"test {best['run_id']} ({best['fuel']}) printed YO2 = {best['Y_O2']}"
                                  + (" (close to the request if Y is a mole fraction; the sheet does not say)" if close
                                     else f" (far from the requested {req}% whichever way Y is defined)"),
                        "quote": f"P = {best['pressure_mmHg']} mmHg, YO2 = {best['Y_O2']}, YN2 = {best['Y_N2']}, YHe = {best['Y_He']}",
                        "location": best["source"]}
        sources = [("PSI metadata", r["derived"]["conditions"][key])]
        pc = r["derived"].get("publication_conditions")
        if pc:
            sources.append(("linked NASA report", pc[key]))
        vals, any_mention = [], None
        for label, cd in sources:
            for m in cd["mentions"]:
                any_mention = any_mention or (label, m)
                for v in m["values"]:
                    x = self._to_unit(v["value"], v["unit"], conv)
                    if x is not None:
                        vals.append((x, v["text"], m, label))
        if not vals and not any_mention:
            return {"status": "not_reported", "detail": f"No {key} information in metadata or linked reports"}
        if not vals:
            label, m = any_mention
            return {"status": "partial", "detail": f"{key} is mentioned ({label}) but no value is given",
                    "quote": m["sentence"], "location": m["location"]}
        tol = {"oxygen": 3.0, "pressure": 15.0, "airflow": 5.0}[key]
        best = min(vals, key=lambda v: abs(v[0] - req))
        st = "match" if abs(best[0] - req) <= tol else "partial"
        return {"status": st, "detail": f"{best[3]} states {best[1]}" + ("" if st == "match" else f" (differs from requested {req})"),
                "quote": best[2]["sentence"], "location": best[2]["location"], "source": best[3]}

    @staticmethod
    def _to_unit(v, unit, conv):
        u = (unit or "").lower()
        if conv is None:
            return v if "%" in u else None
        if conv == "kpa":
            return {"kpa": v, "pa": v / 1000, "atm": v * 101.325, "bar": v * 100, "psia": v * 6.89476, "psi": v * 6.89476,
                    "torr": v * 0.133322}.get(u)
        if conv == "cms":
            return {"cm/s": v, "mm/s": v / 10, "m/s": v * 100}.get(u)
        return None

    # ------------------------------------------------------------ sensors / measurement types
    SENSORS = [
        ("thermocouples", r"thermocouple"), ("radiometers", r"radiometer"),
        ("colour / video cameras", r"(?:colou?r|video|digital|still|hd) camera|video imag|camcorder"),
        ("infrared / UV imaging", r"infrared camera|\bIR camera|ultraviolet|\bUV\b|\bLLUV\b"),
        ("CH* / OH* chemiluminescence imaging", r"\bCH\*|\bOH\*|chemiluminescen"),
        ("gas analyzers (O2, CO, CO2)", r"CSA-CP|compound specific analy[sz]er|carbon dioxide monitor|\bCDM\b|oxygen sensor|gas chromatograph|gas analy[sz]er"),
        ("pressure transducers", r"pressure transducer|pressure sensor"),
        ("flow / anemometer", r"anemometer|flow ?meter|mass flow controller"),
        ("particle / smoke instruments", r"P-Trak|DustTrak|thermal precipitator|particle counter|aerosol monitor|smoke detector|ionization detector|photoelectric"),
        ("soot / laser diagnostics", r"laser extinction|soot volume fraction|\bLII\b|laser-induced|two-colou?r pyrometry|pyrometr"),
        ("accelerometers (g-level)", r"accelerometer|\bSAMS\b|acceleration environment"),
        ("droplet / flame size imaging", r"droplet diameter|flame diameter|flame standoff|backlit"),
    ]

    def sensors_for(self, rid):
        if not hasattr(self, "_sensor_cache"):
            self._sensor_cache = {}
        if rid in self._sensor_cache:
            return self._sensor_cache[rid]
        found = {}
        for i in self.chunks_by_record.get(rid, []):
            c = self.chunks[i]
            if c.get("source_kind") == "psi_document" or c.get("link_level", "").startswith("family"):
                continue
            for name, pat in self.SENSORS:
                if name not in found and re.search(pat, c["text"], re.I):
                    found[name] = {"sensor": name, "sentence": c["text"][:300], "location": c["location"], "url": c["url"],
                                   "source_kind": c.get("source_kind")}
        self._sensor_cache[rid] = list(found.values())
        return self._sensor_cache[rid]

    # ------------------------------------------------------------ fire-safety map
    PILLARS = [("prevention_material_flammability", "Prevention: material flammability"),
               ("detection_smoke", "Detection: smoke and aerosols"),
               ("suppression_extinguishment", "Suppression and extinguishment"),
               ("supporting_combustion_science", "Supporting combustion science")]
    EARTH_RE = re.compile(r"\bon earth\b|terrestrial|earth benefit|industr|firefight|building|residential|consumer|commercial|"
                          r"smoke detector|engine|power generation|emission|pollut|wildfire|fire safety on earth|everyday", re.I)

    def safety_map(self):
        comb = [r for r in self.records if self.val(r, "research_area") == COMBUSTION]
        pillars = []
        for tag, label in self.PILLARS:
            items = []
            for r in comb:
                t = next((t for t in r["derived"]["intent_tags"] if t["tag"] == tag), None)
                if not t:
                    continue
                ev = t["evidence"][0]
                items.append({"id": r["id"], "acronym": self.val(r, "acronym") or r["id"], "title": self.val(r, "title"),
                              "context_only": (t.get("review") or {}).get("verdict") == "partial",
                              "hardware": self.val(r, "hardware") or NR, "gravity": r["derived"]["gravity_tested"]["value"],
                              "n_runs": len(self.runs_by_rec.get(r["id"], [])), "n_reports": len(r.get("literature", [])),
                              "n_data_files": len(r.get("data_files", [])),
                              "evidence": ev["sentence"], "location": ev["location"], "url": r["url"]})
            items.sort(key=lambda x: (x["context_only"], -(x["n_runs"] + x["n_data_files"]), -x["n_reports"], x["acronym"]))
            pillars.append({"tag": tag, "label": label, "items": items})
        earth = []
        for r in comb:
            fd = r["fields"].get("research_impacts", {})
            if fd.get("status") == NR:
                continue
            from_sent = [s for s in re.split(r"(?<=[.!?])\s+", " ".join(fd["value"].split())) if self.EARTH_RE.search(s) and len(s) > 40]
            if from_sent:
                earth.append({"id": r["id"], "acronym": self.val(r, "acronym") or r["id"], "sentence": from_sent[0][:400],
                              "location": fd["source"]["location"], "url": r["url"]})
        hw = {}
        for r in comb:
            h = self.val(r, "hardware") or "not reported"
            hw.setdefault(h, []).append(self.val(r, "acronym") or r["id"])
        return {"pillars": pillars, "earth_applications": earth,
                "hardware_families": [{"hardware": k, "investigations": v} for k, v in sorted(hw.items(), key=lambda kv: -len(kv[1]))],
                "note": "Pillars come from keyword tags reviewed by the AI agent (incorrect tags removed; 'context only' = the topic is motivation, "
                        "not the investigation's aim). Earth applications are sentences quoted from each PSI record's "
                        "'Research Impacts/Earth Benefits' field."}

    # ------------------------------------------------------------ variables / export
    def run_variables(self, inv):
        """What varied across an investigation's runs and what stayed the same (from the experimental table)."""
        rs = self.runs_by_rec.get(inv, [])
        if not rs:
            return None

        def num(u, key, sub="value"):
            v = u.get(key)
            return v.get(sub) if isinstance(v, dict) and v.get("status") == "reported" else None
        rows = []

        def add(label, vals, unit=""):
            vals = [v for v in vals if v not in (None, "", NR)]
            if not vals:
                rows.append({"variable": label, "status": "not reported", "detail": ""})
                return
            if all(isinstance(v, (int, float)) for v in vals):
                lo, hi = min(vals), max(vals)
                rows.append({"variable": label, "status": "varied" if hi > lo else "constant",
                             "detail": f"{lo}–{hi} {unit}".strip() if hi > lo else f"{lo} {unit}".strip(), "n": len(vals)})
            else:
                c = {}
                for v in vals:
                    c[v] = c.get(v, 0) + 1
                rows.append({"variable": label, "status": "varied" if len(c) > 1 else "constant",
                             "detail": ", ".join(f"{k} ({n})" for k, n in sorted(c.items(), key=lambda kv: -kv[1])[:8]), "n": len(vals)})
        add("material", [u.get("material_class", "").replace(" (per SRD manifest)", "") for u in rs])
        add("geometry", [u.get("geometry") for u in rs])
        add("thickness", [num(u, "thickness_mm") for u in rs], "mm")
        add("rod diameter", [num(u, "rod_diameter_mm") for u in rs], "mm")
        add("initial O2", [num(u, "o2_initial") for u in rs], "%")
        add("air flow (first setting)", [num(u, "airflow") for u in rs], "cm/s")
        add("flow direction / configuration", [u.get("flow_direction") or u.get("flow_configuration") for u in rs])
        add("reported ignition outcome", [(u.get("outcome") or {}).get("ignition") for u in rs if (u.get("outcome") or {}).get("status") == "reported"])
        add("how the burn ended", [(u.get("outcome") or {}).get("termination") for u in rs if (u.get("outcome") or {}).get("status") == "reported"])
        add("gravity", [u.get("gravity") for u in rs])
        return {"investigation": inv, "acronym": rs[0]["acronym"], "n_runs": len(rs), "rows": rows}

    CSV_COLS = ["investigation", "acronym", "run_id", "date", "material_text", "material_class", "geometry", "thickness_mm",
                "rod_diameter_mm", "o2_initial_pct", "o2_final_pct", "co2_rise_pct_vol", "co_initial_ppm", "co_final_ppm",
                "airflow_first_cm_s", "airflow_sequence_cm_s", "flow_direction_or_config", "ug_spread_rate_mm_s",
                "weight_loss_rate_mg_min", "outcome_ignition", "outcome_termination", "outcome_observation_verbatim",
                "outcome_source", "flags", "source_file", "source_line"]

    def runs_csv(self, investigation=""):
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.CSV_COLS)
        g = lambda u, k: (u.get(k) or {}).get("value", "") if isinstance(u.get(k), dict) and u[k].get("status") == "reported" else ""
        for u in self.runs:
            if investigation and u["investigation"] != investigation:
                continue
            oc = u.get("outcome") or {}
            af = u.get("airflow") or {}
            w.writerow([u["investigation"], u["acronym"], u["run_id"], u.get("date", ""), u.get("material_text", ""), u.get("material_class", ""),
                        u.get("geometry", ""), g(u, "thickness_mm"), g(u, "rod_diameter_mm"), g(u, "o2_initial"), g(u, "o2_final"),
                        u.get("co2_rise_pct_vol", "") if isinstance(u.get("co2_rise_pct_vol"), float) else "",
                        g(u, "co_initial"), g(u, "co_final"), g(u, "airflow"),
                        " ".join(str(x) for x in af.get("sequence", [])), u.get("flow_direction") or u.get("flow_configuration", ""),
                        g(u, "ug_spread_rate_mm_s"), g(u, "weight_loss_rate_mg_min"),
                        oc.get("ignition", "") if oc.get("status") == "reported" else "",
                        oc.get("termination", "") if oc.get("status") == "reported" else "",
                        oc.get("observation_verbatim", "") if oc.get("status") == "reported" else "",
                        (oc.get("source") or {}).get("location", ""), " | ".join(u.get("flags", [])),
                        u["source"]["file"], u["source"]["line"]])
        return buf.getvalue()

    # ------------------------------------------------------------ test runs
    def runs_query(self, investigation="", material="", o2min=None, o2max=None, flagged=""):
        out = []
        for u in self.runs:
            if investigation and u["investigation"] != investigation:
                continue
            if material and material.lower() not in ((u.get("material_text") or "") + " " + (u.get("material_class") or "")).lower():
                continue
            o2 = u.get("o2_initial", {})
            if (o2min is not None or o2max is not None):
                if o2.get("status") != "reported":
                    continue
                if o2min is not None and o2["value"] < o2min:
                    continue
                if o2max is not None and o2["value"] > o2max:
                    continue
            if flagged == "1" and not u.get("flags"):
                continue
            out.append({k: v for k, v in u.items() if k != "table_notes"})
        o2v = [u["o2_initial"]["value"] for u in self.runs if u.get("o2_initial", {}).get("status") == "reported"]
        # descriptive outcome summary over the filtered BASS-II runs (no model, no extrapolation)
        oc = [u for u in out if (u.get("outcome") or {}).get("status") == "reported"]
        groups = {}
        for u in oc:
            g = f"{u.get('material_class', NR).replace(' (per SRD manifest)', '')} · {u.get('geometry', NR)}"
            groups.setdefault(g, []).append(u)
        outcome_rows = []
        for g, us in sorted(groups.items()):
            sus = [u for u in us if u["outcome"]["ignition"] == "sustained" and u["o2_initial"].get("status") == "reported"]
            nos = [u for u in us if u["outcome"]["ignition"] == "not_sustained" and u["o2_initial"].get("status") == "reported"]
            term = {}
            for u in us:
                term[u["outcome"]["termination"]] = term.get(u["outcome"]["termination"], 0) + 1
            lo = min(sus, key=lambda u: u["o2_initial"]["value"]) if sus else None
            hi_n = max(nos, key=lambda u: u["o2_initial"]["value"]) if nos else None
            outcome_rows.append({"group": g, "n": len(us), "sustained": sum(1 for u in us if u["outcome"]["ignition"] == "sustained"),
                                 "not_sustained": sum(1 for u in us if u["outcome"]["ignition"] == "not_sustained"),
                                 "terminations": term,
                                 "lowest_o2_with_sustained_flame": {"run": lo["run_id"], "o2": lo["o2_initial"]["value"]} if lo else None,
                                 "highest_o2_without_sustained_flame": {"run": hi_n["run_id"], "o2": hi_n["o2_initial"]["value"],
                                                                        "observation": hi_n["outcome"]["observation_verbatim"][:200]} if hi_n else None})
        invs = sorted({u["investigation"] for u in out})
        return {"n": len(out), "runs": out, "tables": self.run_tables, "outcome_summary": outcome_rows,
                "variables": [v for v in (self.run_variables(i) for i in invs) if v],
                "outcome_note": "Counts of reported outcomes in BASS-II (microgravity, ISS) for the filtered runs. Outcomes are quoted from "
                                "NASA/TM-20210011385 Appendix A and labelled by the AI agent (not a human review). Ignition failures often have "
                                "procedural causes (reused sample, igniter angle, flat tip) stated in the quote, so these counts are not "
                                "flammability limits or ignition probabilities.",
                "summary": {"total_runs": len(self.runs), "with_measured_initial_o2": len(o2v),
                            "o2_range": [min(o2v), max(o2v)] if o2v else None,
                            "by_investigation": {t["investigation"]: t["runs"] for t in self.run_tables}},
                "note": "Rows come straight from PSI 'Experimental table' files. Raw cells are kept; parsed values appear only when unambiguous. "
                        "Flags mark suspect or ambiguous cells. Gravity regime is per investigation and is not interchangeable."}

    # ------------------------------------------------------------ overview / coverage
    def overview(self):
        comb = [r for r in self.records if self.val(r, "research_area") == COMBUSTION]
        snap = (self.manifest or {}).get("catalog_snapshot", {})
        m = self.metrics or {}
        dep = (m.get("results", {}).get(m.get("deployed_model", "softmax_regression"), {}) or {}).get("summary", {})
        return {
            "n_records": len(self.records), "n_combustion": len(comb),
            "n_fire_safety_subarea": sum(1 for r in comb if "fire" in self.val(r, "subresearch_area").lower()),
            "n_local_archives": sum(1 for r in self.records if r["local_files"]),
            "n_chunks": len(self.chunks),
            "n_with_measurements_inspected": sum(1 for r in self.records if r.get("data_files")),
            "n_data_files": sum(len(r.get("data_files", [])) for r in self.records),
            "n_runs": len(self.runs), "n_run_tables": len(self.run_tables),
            "n_outcomes": sum(1 for u in self.runs if (u.get("outcome") or {}).get("status") == "reported"),
            "n_reports": sum(len(r.get("literature", [])) for r in self.records),
            "n_records_with_reports": sum(1 for r in self.records if r.get("literature")),
            "n_report_chunks": sum(1 for c in self.chunks if c.get("source_kind") == "ntrs_report"),
            "catalog_snapshot": snap, "facets": self.facets(),
            "local_archives": [{"file": a["file"], "id": a.get("internal_study_identifier"),
                                "acronym": (a.get("official_record") or {}).get("official_acronym"),
                                "sha256": a.get("sha256"), "mapping_status": a.get("mapping_status"),
                                "mapping_note": a.get("mapping_note")} for a in (self.manifest or {}).get("archives", [])],
            "model": {"available": bool(self.model_info), "cv_accuracy": dep.get("accuracy"), "cv_macro_f1": dep.get("macro_f1"),
                      "cv_relevance_accuracy": dep.get("combustion_accuracy"),
                      "n_training": m.get("n_investigations")},
            "conflicts": (self.ingest_log or {}).get("counts", {}).get("conflicts"),
        }

    def coverage(self, scope="combustion"):
        pool = [r for r in self.records if scope != "combustion" or self.val(r, "research_area") == COMBUSTION]
        n = len(pool)
        conds = {}
        for k in ("oxygen", "pressure", "airflow"):
            ment = [r["id"] for r in pool if r["derived"]["conditions"][k]["status"] != "not_extracted"]
            num = [r["id"] for r in pool if r["derived"]["conditions"][k]["has_numeric_value"]]
            conds[k] = {"denominator": n, "mentioned": len(ment), "with_numeric_value": len(num), "ids_with_value": num,
                        "not_extracted": n - len(ment)}
        grav_t, grav_m = {}, {}
        for r in pool:
            g = r["derived"]["gravity_tested"]["value"]
            grav_t[g] = grav_t.get(g, 0) + 1
            for k, v in r["derived"]["gravity_mentions"].items():
                if v:
                    grav_m[k] = grav_m.get(k, 0) + 1
        mats = {}
        for r in pool:
            for m in r["derived"]["materials"]:
                mats[m["material"]] = mats.get(m["material"], 0) + 1
        fields = {}
        for f in ("objectives", "approach", "hypothesis", "research_impacts", "hardware", "flight_platform", "start_date", "subresearch_area"):
            fields[f] = sum(1 for r in pool if self.val(r, f))
        for k in ("oxygen", "pressure", "airflow"):
            pub = [r["id"] for r in pool if (r["derived"].get("publication_conditions") or {}).get(k, {}).get("has_numeric_value")]
            either = [r["id"] for r in pool if r["id"] in pub or r["derived"]["conditions"][k]["has_numeric_value"]]
            conds[k].update({"with_numeric_value_in_linked_report": len(pub), "with_numeric_value_any_source": len(either)})
        lit = {"records_with_linked_reports": sum(1 for r in pool if r.get("literature")),
               "records_with_report_full_text": sum(1 for r in pool if any(l.get("text_indexed", "").startswith("full") for l in r.get("literature", []))),
               "reports_linked": sum(len(r.get("literature", [])) for r in pool)}
        ids = {r["id"] for r in pool}
        pr = [u for u in self.runs if u["investigation"] in ids]
        o2r = [u for u in pr if u.get("o2_initial", {}).get("status") == "reported"]
        runs_cov = {"investigations_with_run_table": len({u["investigation"] for u in pr}), "runs": len(pr),
                    "runs_with_measured_o2": len(o2r),
                    "o2_range": [min(u["o2_initial"]["value"] for u in o2r), max(u["o2_initial"]["value"] for u in o2r)] if o2r else None,
                    "runs_with_airflow_value": sum(1 for u in pr if u.get("airflow", {}).get("status") == "reported"),
                    "runs_flagged": sum(1 for u in pr if u.get("flags") and not all(f.startswith("O2 'Derived") for f in u["flags"]))}
        return {"scope": scope, "denominator": n, "conditions": conds, "gravity_tested": grav_t, "literature": lit, "runs": runs_cov,
                "gravity_mentioned_in_text": grav_m, "materials": dict(sorted(mats.items(), key=lambda kv: -kv[1])),
                "field_completeness": fields,
                "data_types": {"metadata_narrative": n, "local_archive_inspected": sum(1 for r in pool if r["local_files"]),
                               "measurements_inspected": sum(1 for r in pool if r.get("data_files")), "data_files_not_inspected": sum(1 for r in pool if not r["local_files"])},
                "note": "Counts describe the indexed PSI catalog metadata only. They do not describe all spacecraft-fire research."}
