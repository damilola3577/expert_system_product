# frontend/app.py
import streamlit as st
import requests
import numpy as np
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="EnergySage — Home Energy Optimizer", layout="wide")
st.title("EnergySage — Home Energy Optimizer (Demo)")

st.sidebar.header("Inputs")
desired = st.sidebar.slider("Desired temperature (°C)", 18, 26, 21)
comfort_w = st.sidebar.slider("Comfort weight (1 low - 5 high)", 1, 5, 2)
outdoor_temp = st.sidebar.slider("Outdoor temp (°C)", -5, 35, 10)
gens = st.sidebar.number_input("GA generations", min_value=10, max_value=500, value=120, step=10)
pop = st.sidebar.number_input("GA population", min_value=10, max_value=200, value=50, step=10)

# Initialize session_state tariff if missing
if "tariff" not in st.session_state:
    base = np.array([0.08] * 24)
    peak_hours = list(range(17, 21))
    base[peak_hours] = 0.28
    st.session_state.tariff = base

def editable_tariff_editor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-version compatibility wrapper for Streamlit data editors.
    Tries experimental_data_editor, then data_editor, otherwise falls back to number_input grid.
    Returns the edited pandas DataFrame.
    """
    # Newer or experimental editors
    if hasattr(st, "experimental_data_editor"):
        try:
            return st.experimental_data_editor(df, num_rows="dynamic")
        except Exception:
            pass
    if hasattr(st, "data_editor"):
        try:
            return st.data_editor(df, num_rows="dynamic")
        except Exception:
            pass

    # Fallback editor: present number_inputs to edit each hour
    st.warning("Interactive table editor not available in this Streamlit version — using fallback editor.")
    edited = df.copy()
    cols = st.columns(6)
    for i in range(24):
        col = cols[i % 6]
        with col:
            val = st.number_input(f"Hour {i}", value=float(df.at[i, "tariff_$per_kwh"]), key=f"tariff_{i}", step=0.01, format="%.4f")
            edited.at[i, "tariff_$per_kwh"] = val
    return edited

st.header("Tariff (24 hourly values) — default demo")
tariff_df = pd.DataFrame({"hour": list(range(24)), "tariff_$per_kwh": st.session_state.tariff})
edited = editable_tariff_editor(tariff_df)

if st.button("Use edited tariff"):
    # Convert to numpy, preserving float dtype
    try:
        st.session_state.tariff = edited["tariff_$per_kwh"].to_numpy(dtype=float)
    except Exception:
        # in case editor returns strange types
        st.session_state.tariff = np.array([float(x) for x in edited["tariff_$per_kwh"].tolist()])
    st.success("Tariff updated")

st.markdown("---")
st.header("Run optimization")
if st.button("Run optimization"):
    payload = {
        "tariff": st.session_state.tariff.tolist(),
        "desired": float(desired),
        "comfort_weight": float(comfort_w),
        "outdoor_temp": float(outdoor_temp),
        "pop_size": int(pop),
        "gens": int(gens)
    }
    with st.spinner("Optimizing..."):
        try:
            r = requests.post(f"{API_BASE}/optimize", json=payload, timeout=300)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
        else:
            res = r.json()
            schedule = res.get("schedule", [])
            total_cost = res.get("total_cost", None)
            comfort_pen = res.get("comfort_penalty", None)
            if schedule:
                st.success(f"Optimization done — estimated daily cost ${total_cost:.2f}")
                st.subheader("Hourly schedule (°C)")
                table = pd.DataFrame({"hour": list(range(len(schedule))), "setpoint_C": schedule})
                st.table(table)
                st.write(f"Comfort penalty (lower better): {comfort_pen:.2f}")
            else:
                st.error("Optimization returned no schedule.")

st.markdown("---")
st.header("Ask for explanation (RAG + LLM demo)")
q = st.text_area("Ask a question about the recommendation (e.g. why raise setpoint at 14:00?)", height=120)
if st.button("Explain"):
    if not q.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Getting explanation..."):
            try:
                r = requests.post(f"{API_BASE}/rag_query", data={"q": q}, timeout=60)
                r.raise_for_status()
            except requests.exceptions.RequestException as e:
                st.error(f"RAG API request failed: {e}")
            else:
                answer = r.json().get("answer", "No answer")
                st.subheader("LLM explanation (demo)")
                st.write(answer)
