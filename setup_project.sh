#!/usr/bin/env bash
set -e

# create branch
git checkout -b feature/energy-sage || git switch -c feature/energy-sage

# create directories
mkdir -p backend frontend knowledge .vscode

# requirements
cat > requirements.txt <<'PYREQ'
fastapi
uvicorn[standard]
streamlit
numpy
pandas
scikit-learn
sentence-transformers
requests
python-multipart
PYREQ

# backend/optimizer.py
cat > backend/optimizer.py <<'PYOPT'
import numpy as np

MIN_SP = 18
MAX_SP = 26
HOURS = 24

def energy_for_setpoint(setpoint, outdoor_temp=10.0):
    base = 0.2
    alpha = 0.08
    return base + alpha * max(0, setpoint - outdoor_temp)

def comfort_penalty(schedule, desired=21):
    diffs = np.abs(schedule - desired)
    weights = 1 / (1 + np.exp(- (diffs - 1.0)))
    return float(np.sum(weights * diffs))

def apply_rules(schedule):
    schedule = np.clip(schedule, MIN_SP, MAX_SP)
    return schedule

def random_population(pop_size):
    return np.random.randint(MIN_SP, MAX_SP + 1, size=(pop_size, HOURS))

def fitness(schedule, tariff, desired=21, outdoor_temp=10.0, comfort_weight=1.0):
    hourly_energy = np.array([energy_for_setpoint(int(sp), outdoor_temp) for sp in schedule])
    cost = np.sum(hourly_energy * tariff)
    penalty = comfort_penalty(schedule, desired)
    return -(cost + comfort_weight * penalty)

def tournament_select(pop, scores, k=3):
    idx = np.random.randint(0, len(pop), k)
    best = idx[np.argmax(scores[idx])]
    return pop[best].copy()

def crossover(a, b):
    point = np.random.randint(1, HOURS-1)
    child = np.concatenate([a[:point], b[point:]])
    return child

def mutate(child, mutation_rate=0.02):
    for i in range(HOURS):
        if np.random.rand() < mutation_rate:
            child[i] = np.random.randint(MIN_SP, MAX_SP+1)
    return child

def run_ga(tariff, pop_size=50, gens=120, desired=21, comfort_weight=1.0, outdoor_temp=10.0):
    pop = random_population(pop_size)
    for g in range(gens):
        scores = np.array([fitness(ind, tariff, desired, outdoor_temp, comfort_weight) for ind in pop])
        new_pop = []
        elite_count = max(1, pop_size // 20)
        elite_idx = np.argsort(scores)[-elite_count:]
        new_pop.extend(pop[elite_idx])
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, scores)
            p2 = tournament_select(pop, scores)
            child = crossover(p1, p2)
            child = mutate(child)
            child = apply_rules(child)
            new_pop.append(child)
        pop = np.array(new_pop)
    scores = np.array([fitness(ind, tariff, desired, outdoor_temp, comfort_weight) for ind in pop])
    best = pop[np.argmax(scores)]
    hourly_energy = np.array([energy_for_setpoint(int(sp), outdoor_temp) for sp in best])
    total_cost = float(np.sum(hourly_energy * tariff))
    comfort = float(comfort_penalty(best, desired))
    return {
        "schedule": best.tolist(),
        "total_cost": total_cost,
        "comfort_penalty": comfort
    }
PYOPT

# backend/rag.py
cat > backend/rag.py <<'PYRAG'
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

class KnowledgeIndex:
    def __init__(self, docs_dir="knowledge"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.docs = []
        self.embeddings = None
        self._load_docs(docs_dir)

    def _load_docs(self, docs_dir):
        docs = []
        if not os.path.exists(docs_dir):
            return
        for fname in os.listdir(docs_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(docs_dir, fname), "r", encoding="utf8") as f:
                    docs.append(f.read())
        self.docs = docs
        if docs:
            self.embeddings = self.model.encode(docs, convert_to_numpy=True)

    def retrieve(self, query, k=3):
        if self.embeddings is None or len(self.docs) == 0:
            return []
        qvec = self.model.encode([query], convert_to_numpy=True)
        nbrs = NearestNeighbors(n_neighbors=min(k, len(self.docs)), metric="cosine").fit(self.embeddings)
        dists, idxs = nbrs.kneighbors(qvec)
        results = []
        for idx in idxs[0]:
            results.append(self.docs[idx])
        return results

def llm_answer(prompt, api_client=None):
    return "LLM answer (demo): Based on retrieved documents and the schedule, it's recommended to raise the setpoint during certain hours to reduce costs while keeping comfort near target."

def rag_query(query, index: KnowledgeIndex, k=3):
    contexts = index.retrieve(query, k=k)
    joined = "\n\n---\n\n".join(contexts)
    prompt = f"User question:\n{query}\n\nContext documents:\n{joined}\n\nPlease answer concisely, reference any specific doc passages if relevant."
    return llm_answer(prompt)
PYRAG

# backend/main.py
cat > backend/main.py <<'PYMAIN'
from fastapi import FastAPI, Form
from pydantic import BaseModel
import numpy as np
from optimizer import run_ga
from rag import KnowledgeIndex, rag_query

app = FastAPI()
index = KnowledgeIndex(docs_dir="knowledge")

class OptimizeRequest(BaseModel):
    tariff: list
    desired: float = 21.0
    comfort_weight: float = 1.0
    outdoor_temp: float = 10.0
    pop_size: int = 50
    gens: int = 120

@app.post("/optimize")
async def optimize(req: OptimizeRequest):
    tariff = np.array(req.tariff, dtype=float)
    res = run_ga(tariff,
                 pop_size=req.pop_size,
                 gens=req.gens,
                 desired=req.desired,
                 comfort_weight=req.comfort_weight,
                 outdoor_temp=req.outdoor_temp)
    return res

@app.post("/rag_query")
async def query(q: str = Form(...)):
    answer = rag_query(q, index, k=3)
    return {"answer": answer}
PYMAIN

# frontend/app.py
cat > frontend/app.py <<'PYFRONT'
import streamlit as st
import requests
import numpy as np
import pandas as pd

API_BASE = "http://localhost:8000"

st.title("EnergySage — Home Energy Optimizer (Demo)")

st.sidebar.header("Inputs")
desired = st.sidebar.slider("Desired temperature (°C)", 18, 26, 21)
comfort_w = st.sidebar.slider("Comfort weight (1 low - 5 high)", 1, 5, 2)
outdoor_temp = st.sidebar.slider("Outdoor temp (°C)", -5, 35, 10)
gens = st.sidebar.number_input("GA generations", min_value=20, max_value=500, value=120, step=10)
pop = st.sidebar.number_input("GA population", min_value=10, max_value=200, value=50, step=10)

st.header("Tariff (24 hourly values) — default demo")
if "tariff" not in st.session_state:
    base = np.array([0.08]*24)
    peak_hours = list(range(17,21))
    base[peak_hours] = 0.28
    st.session_state.tariff = base

tariff_df = pd.DataFrame({"hour": list(range(24)), "tariff_$per_kwh": st.session_state.tariff})
edited = st.experimental_data_editor(tariff_df, num_rows="dynamic")
if st.button("Use edited tariff"):
    st.session_state.tariff = edited["tariff_$per_kwh"].to_numpy()
    st.success("Tariff updated")

if st.button("Run optimization"):
    payload = {
        "tariff": st.session_state.tariff.tolist(),
        "desired": desired,
        "comfort_weight": comfort_w,
        "outdoor_temp": outdoor_temp,
        "pop_size": pop,
        "gens": gens
    }
    with st.spinner("Optimizing..."):
        r = requests.post(f"{API_BASE}/optimize", json=payload, timeout=120)
    if r.status_code == 200:
        res = r.json()
        schedule = res["schedule"]
        total_cost = res["total_cost"]
        comfort_pen = res["comfort_penalty"]
        st.success(f"Optimization done — estimated daily cost ${total_cost:.2f}")
        st.subheader("Hourly schedule (°C)")
        table = pd.DataFrame({"hour": list(range(24)), "setpoint_C": schedule})
        st.table(table)
        st.write(f"Comfort penalty (lower better): {comfort_pen:.2f}")
    else:
        st.error("API error: " + r.text)

st.header("Ask for explanation (RAG + LLM demo)")
q = st.text_area("Ask a question about the recommendation (e.g. why raise setpoint at 14:00?)")
if st.button("Explain"):
    with st.spinner("Getting explanation..."):
        r = requests.post(f"{API_BASE}/rag_query", data={"q": q})
    if r.status_code == 200:
        st.subheader("LLM explanation (demo)")
        st.write(r.json().get("answer", "No answer"))
    else:
        st.error("Error from backend")
PYFRONT

# knowledge/example files
cat > knowledge/hvac_basics.txt <<'TXT'
Heating energy scales with the difference between indoor setpoint and outdoor temperature.
Small changes (1-2°C) can yield meaningful energy savings over extended periods.
TXT

cat > knowledge/load_shifting.txt <<'TXT'
Shift flexible loads like EV charging, dishwasher, and laundry to off-peak hours to reduce cost.
TXT

cat > knowledge/tariff_faq.txt <<'TXT'
Time-of-use tariffs charge more during peak hours. Check your utility's TOU schedule for exact hours and rates.
TXT

# README update
cat > README.md <<'README'
# EnergySage - Expert System Product (Student Demo)

This repo contains the EnergySage demo: a hybrid expert system (GA + fuzzy + rules) and a RAG-powered explanation pipeline with a simple Streamlit front-end.

See README in the repo root for run instructions.
README

# .gitignore
cat > .gitignore <<'GITIGN'
.venv
__pycache__
*.pyc
*.pkl
GITIGN

# add and commit
git add .
git commit -m "Add EnergySage demo (backend, frontend, knowledge, requirements)"
echo "Setup complete on branch feature/energy-sage. Run 'git push origin feature/energy-sage' to push."