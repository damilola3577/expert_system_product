import streamlit as st
import requests
import numpy as np
import pandas as pd

API_BASE = "http://localhost:8000"

st.title("EnergySage — Home Energy Optimizer (Demo)")

st.sidebar.header("Inputs")
desired = st.sidebar.slider("Desired temperature (°C)", 18, 26, 21)
comfort_w = st.sidebar.slider("Comfort weight (1 low - 5 high)", 1, 5, 2)
outdoor_temp = st.sidebar.slider("Outdoor temp (C)", -5, 35, 10)
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
        st.success(f"Optimization done  estimated daily cost ${total_cost:.2f}")
        st.subheader("Hourly schedule (C)")
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
