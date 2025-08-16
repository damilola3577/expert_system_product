# EnergySage — Smart Home Energy Optimizer

﻿# Expert System Product: EnergySage — Smart Home Energy Optimizer 
Contains backend (FastAPI), frontend (Streamlit), and knowledge files for a RAG demo.

## Overview
EnergySage is a hybrid expert system that uses a genetic algorithm + fuzzy comfort rules to generate cost- and comfort-optimized HVAC schedules, and a RAG-augmented fine-tuned LLM to explain recommendations and provide personalized energy-saving advice. It is a demo-grade, end-to-end prototype that finds a cost-comfort balance for home HVAC, EV charging, solar and storage scheduling. The app combines a lightweight Streamlit front end with a FastAPI backend that runs a hybrid optimizer (Genetic Algorithm + fuzzy comfort rules + hard safety rules) and a Retrieval-Augmented Generation (RAG) layer that provides document-grounded explanations via a fine-tuned LLM or local TF–IDF fallback.

This README section documents the project layout, how to run the demo, representative performance numbers, and important caveats for production.

---

## Quickstart (local demo)

**Prerequisites**

* Python 3.9+ (venv recommended)
* Install dependencies: `pip install -r requirements.txt`

**Run the backend**

```bash
# from project root
uvicorn app.main:app --reload --port 8000
```

**Run the front-end (Streamlit)**

```bash
streamlit run app/ui/streamlit_app.py
```

Open `http://localhost:8501` for the UI and `http://localhost:8000/docs` for FastAPI OpenAPI docs.

**Example API calls**

```bash
# Optimize
curl -X POST http://localhost:8000/optimize -H "Content-Type: application/json" \
  -d '{"tariff": [0.08,...], "desired_temp": 21, "comfort_weight": 3, "outdoor_temp": 5}'

# RAG query (explain)
curl -X POST http://localhost:8000/rag_query -H "Content-Type: application/json" \
  -d '{"query": "Why was EV charging scheduled at 3:00?", "context": {...}}'
```

---

## Demo scenario & representative numbers

> These are demo numbers for a typical laptop/desktop CPU. Replace with measurements from your dataset and hardware.

* **Tariff**: peak (17:00–20:00) = \$0.28/kWh, off-peak = \$0.08/kWh
* **GA (demo)**: `pop_size=50`, `gens=120` (tune lower for quick demos)
* **Energy model**: linear proxy for heating energy per °C above outdoor temperature

**Observed example (illustrative)**

* Runtime: \~20–90 s depending on `gens`/`pop` and CPU. For short demos use `gens=30`, `pop=30` → \~10–30 s.
* Baseline (flat 21°C) daily cost: **\$6.20**
* Optimized daily cost: **\$4.35** → **\~30%** daily cost reduction (example)
* Comfort penalty: small (example: **8.4**) — \~1°C average deviation
* Explain latency: local TF–IDF + stub LLM: **< 1 s**. Remote fine-tuned LLM adds API latency (100s ms → seconds).

---

## Front-end (Streamlit)

**Main elements**

* Sidebar: desired temperature (18–26°C), comfort weight (1–5), outdoor temp (−5–35°C), GA parameters (generations, population)
* Main area: 24-hour tariff editor, Run optimization button, results panel (schedule, cost, comfort penalty), charts, and an Explain box that calls `/rag_query`.

**User flow**

1. Edit or upload hourly tariff (or use demo default).
2. Pick comfort tradeoff and desired setpoint.
3. Click **Run optimization** → display schedule + metrics.
4. Optionally ask “Why this change?” → receives a document-referenced plain-language explanation.

---

## Backend & algorithms

* **Rule engine**: enforces hard safety constraints (min/max setpoints) at input and solution stages.
* **Fuzzy comfort score**: small deviations incur small penalties; larger deviations scale non-linearly. Used in fitness.
* **Genetic Algorithm**: 24-hour schedule = chromosome. Fitness = `-(cost + comfort_penalty)`. GA uses selection, crossover, mutation, and elitism.
* **RAG layer**: local TF–IDF or sentence-embedding retriever searches `knowledge/` for supporting passages. Retrieved snippets + optimizer outputs are supplied to a fine-tuned LLM for human-friendly explanations.

---

## Data & knowledge sources

* `knowledge/` — device manuals, tips, policy notes, and best-practice text used by the retriever.
* Optional: persist semantic vectors in FAISS for improved retrieval at scale.

---

## Notes & caveats

* The energy model is illustrative. Integrate a physics-based thermal model or real HVAC power model for production-grade savings.
* GA runtime scales roughly linearly with `pop_size * gens`: tune these for demo speed vs. solution quality.
* RAG quality improves with semantic embeddings and a production vector DB (FAISS) + fine-tuned LLM.

---

## Safety & limitations

* Hard rules prevent unsafe temperatures. Always add UI disclaimers for medically vulnerable occupants.
* Current limitations: simplified energy model, demo LLM stub, and TF–IDF fallback for retrieval.

---

## Next steps

1. Integrate a thermal/HVAC power model for accurate savings.
2. Persist semantic vectors with FAISS and add scheduled re-indexing.
3. Fine-tune an LLM on domain-specific energy-explanation dialogues.
4. Add user authentication, telemetry, and remote deployment.

---

## Contributing & contact

Contributions welcome. Please open an issue or PR with proposed changes. For discussion or research collaboration, contact the repo owner.

---

*This README section is intended as a repo-ready summary and quick reference for developers and reviewers.*
