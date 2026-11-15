---
title: Flame in Freefall
emoji: 🔥
colorFrom: red
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Evidence explorer for NASA microgravity fire research
---

# Flame in Freefall

An evidence-first web app for finding, comparing and understanding NASA Physical Sciences Informatics (PSI) combustion investigations that bear on spacecraft fire safety, including **146 measured test runs** and **125 test outcomes quoted from NASA reports**. Built for the NASA Space Apps Challenge.

> **This is a research discovery and evidence-assessment tool.** It is not a certified fire-risk calculator, flight rule or operational safety system. Relevance scores measure how well indexed evidence matches a query. They are not fire risk, ignition probability, or proof that any material is safe or unsafe.

## Quick start (Windows, macOS, Linux)

Running the app requires **Python 3.10+** and nothing else: no packages, no Node, no API keys.

```bash
cd nasa_working
python app/server.py
```

Then open <http://127.0.0.1:8765>. The server binds to localhost only and makes no outbound network calls. All processed data and the trained model ship in `data/processed/` and `ml/artifacts/`, so the app runs straight away.

Rebuilding from the PDFs additionally needs `pip install pypdf` (text extraction only; the extracted text is already cached in `data/processed/ntrs_text/`).

## Rebuild from sources

```bash
python run_all.py            # audit -> ingest -> train -> ranking eval -> review sheets -> tests  (~6 min)
python run_all.py --quick    # 1-seed training, skips the ranking evaluation  (~1 min)
python run_all.py --fetch    # also re-download the PSI catalog and public NTRS reports (needs internet)
```

| Step | Command | Output |
|---|---|---|
| Fetch the official PSI catalog (metadata) | `python pipeline/fetch_psi_catalog.py` | `data/raw/psi_api/*.json`, `fetch_log.jsonl` |
| Audit the six original ZIPs | `python pipeline/audit_sources.py` | `data/raw/source_manifest.json` |
| Find and download public NASA reports (NTRS) | `python pipeline/fetch_ntrs.py` | `data/raw/ntrs/` |
| Ingest everything with provenance (see below) | `python pipeline/ingest.py` | `data/processed/records.json`, `chunks.json`, `runs.json`, `bass2_outcomes.json`, `ingest_log.json` |
| Train and evaluate the model | `python ml/train.py` | `ml/artifacts/research_area_model.json`, `metrics.json`, `oof_predictions.json` |
| Proxy ranking evaluation | `python ml/eval_ranking.py` | `ml/artifacts/ranking_eval.json` |
| Review sheets (intent tags, BASS-II outcomes) | `python ml/review_intent_tags.py`, `python ml/review_bass2_outcomes.py` | `ml/review/*.csv` |
| Tests | `python -m unittest discover -s tests -v` | 29 checks |

`ingest.py` runs the following modules in order:
1. `literature.py`: NASA report text with page numbers.
2. `psi_files.py`: PSI data files.
3. `bass2_outcomes.py`: BASS-II outcomes from NASA TM-20210011385.
4. `run_tables.py`: PSI experimental tables, which become test runs.

## What is inside

| Layer | Content | Provenance |
|---|---|---|
| Catalog | 133 PSI investigations (24 Combustion Science), plus the six local ZIP archives cross-checked field by field (0 conflicts) | PSI record and field, or ZIP and line |
| Public NASA reports | 68 NTRS links on 15 combustion investigations, 28 PDFs indexed (11,906 sentences) | NTRS id and page |
| **Measured test runs** | 146 runs from PSI "Experimental table" files: BASS-II 129, SAFFIRE-II 9, SAME 8 (O2, CO2, CO, airflow, material, thickness, spread rate…) | file and line; BASS-II cross-checked against the PSI spreadsheet (129/129 consistent) |
| **Test outcomes** | 125 BASS-II outcomes quoted verbatim from NASA TM-20210011385 (Appendix A, Tables 5.1 and 7.1), labelled ignition and how the burn ended | NTRS id and page |
| PSI data files (FLEX) | chamber pressure/temperature log, flame-diameter time series (FLEX-187), run sheet (FLEX-263), camera timing logs | file and column/page |
| Trained model | TF-IDF and softmax-regression classifier trained from scratch on the 133 PSI records plus 673 NASA Task Book Physical Sciences projects (NASA discipline labels aligned to PSI areas; leakage-filtered per fold). Evaluated on PSI only, with grouped CV: **100% fire-safety relevance accuracy** (baseline 82.0%); **6-area accuracy 0.945, macro-F1 0.939** (PSI-only training: 0.852 and 0.691) | `docs/MODEL_CARD.md` |

**App views:**
- **Overview, Research explorer and Investigation detail:** explained ranking; every field cited.
- **Test runs:** measured runs, quoted outcomes and an outcome summary.
- **Compare:** "Not reported" is never filled in.
- **Evidence assistant:** extractive. It builds a cited profile for one named investigation and a cited comparison when several are named, and it abstains on weak evidence.
- **Scenario evidence explorer:** matches against measured runs first and never substitutes one gravity regime for another.
- **Fire safety map:** investigations grouped by prevention, detection and suppression; hardware families; Earth applications quoted from PSI "Research Impacts/Earth Benefits".
- **Crew-environment presets** in the scenario explorer: ISS, Moon Base, pressurized rover (≥30% O2 at ≤10.2 psi, lunar gravity), Mars base and Earth. They surface elevated-oxygen evidence (for example ACME BRE at 10.2 psia/26.5% and 8.2 psia/34% O2) and the micro-/normal-gravity bracketing note.
- **Per investigation:** sensors and measurement types (cited), and "what varied versus what stayed the same" across runs.
- **Data export for modelling:** `/api/runs.csv` and `/api/runs.json` (buttons on the Test runs tab), with source file and line, flags and outcome quotes on every row.
- **Coverage and gaps, Methodology, Demo.**

## Demo flow (about 3 minutes)

Open the **Demo** tab. Each step has a button that runs it on real data:

1. **Overview:** live counts.
2. **Explorer:** search *"flame spread over fabric with opposed airflow"* to see matched terms and quoted evidence.
3. **Test runs:** BASS-II runs with measured O2/CO2/CO and the NASA-reported outcome for each. Nomex kept no flame (0 of 3) even at about 22% O2.
4. **FLEX measured data:** chamber log, flame diameter over time, and the FLEX-263 run sheet.
5. **Assistant:** *"What was the extinction diameter in FLEX-263?"* (1.421 mm, from the run sheet), and a cited comparison of BASS, SAFFIRE, SAME and FLEX.
6. **Scenario:** a lunar habitat at 30% O2. No indexed run went above 22.2% O2, and none was conducted in lunar gravity, so the tool says the evidence is unsupported. At 17% O2, it finds 86 matching measured runs.
7. **Coverage and Methodology:** denominators, model metrics and limits.

## Project layout

```
nasa_working/
  app/            server.py (stdlib HTTP + JSON API), engine.py (search, ranking, assistant, scenario, runs), static/ (UI)
  pipeline/       fetch_psi_catalog.py, fetch_ntrs.py, audit_sources.py, isa.py, extract.py, ingest.py, literature.py,
                  psi_files.py, run_tables.py, bass2_outcomes.py, xlsx_lite.py, common.py
  ml/             textml.py, train.py, eval_ranking.py, review_*.py, artifacts/, review/
  data/raw/       psi_api/ (catalog snapshots), ntrs/ (reports), psi_files/<PSI-ID>/ (owner-supplied PSI files, unchanged)
  data/processed/ records, chunks, runs, bass2_outcomes, ntrs_text cache, model dataset
  docs/           MODEL_CARD.md, SOURCE_REGISTER.md, FINAL_REPORT.md
  tests/          test_core.py
```

The original ZIPs stay in the parent folder and are opened read-only. The tests check that their SHA-256 hashes are unchanged.

## Known limitations

- **No fire-outcome prediction.** We deliberately trained no ignition or extinction model. Only 15 BASS-II tests had no sustained flame, often for procedural reasons stated in the report, and how a burn ended mostly reflects the operator's procedure. See the Methodology tab.
- **Outcome labels are AI-reviewed.** The quotes are verbatim, but the ignition and termination labels and the intent tags were assigned by the AI coding agent at the owner's request, not by a human or PI.
- **Measured runs cover three investigations.** These are BASS-II, SAFFIRE-II and SAME, plus FLEX data files. Other investigations rely on metadata and report sentences. No run exceeds 22.2% O2, and none was conducted in lunar or Martian gravity.
- **The model gives no ranking benefit.** On 95 proxy queries, BM25 plus the model scores the same as BM25 alone (nDCG@10 0.636 vs 0.640). The model is an explainable topical classifier.
- **The catalog is a 2026-11-13 snapshot.** The original project spec file was not available, so the "26 results" search was reconciled from the live PSI catalog instead (see `docs/SOURCE_REGISTER.md`).
