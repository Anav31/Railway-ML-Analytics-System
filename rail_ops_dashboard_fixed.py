import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, os, random

# ---------------------------- Config ----------------------------
st.set_page_config(
    page_title="RailOps — Indian Railways Control Dashboard", 
    page_icon="🚆", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF9933;
        font-weight: 700;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #138808;
        margin-bottom: 10px;
    }
    .train-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF9933;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background-color: #FF9933;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Indian Railways stations data
INDIAN_STATIONS = {
    "NDLS": {"name": "New Delhi", "zone": "NR"},
    "BCT": {"name": "Mumbai Central", "zone": "WR"},
    "HWH": {"name": "Howrah Junction", "zone": "ER"},
    "MAS": {"name": "Chennai Central", "zone": "SR"},
    "SBC": {"name": "Bengaluru City", "zone": "SWR"},
}

# ---------------------------- Sample trains ----------------------------
def sample_trains():
    now = datetime.now().replace(second=0, microsecond=0)
    
    base = [
        (12001, "New Delhi - Bhopal Shatabdi", "Shatabdi", "NDLS", "BCT", 90),
        (12951, "Mumbai Rajdhani", "Rajdhani", "BCT", "NDLS", 95),
        (12259, "Sealdah Duronto", "Duronto", "NDLS", "HWH", 85),
        (12675, "Kovai Express", "Express", "MAS", "SBC", 60),
        (12301, "Howrah Rajdhani", "Rajdhani", "HWH", "NDLS", 92),
    ]
    
    rows = []
    for tid, name, ttype, origin, dest, speed in base:
        origin_info = INDIAN_STATIONS[origin]
        dest_info = INDIAN_STATIONS[dest]
        
        sched_dep = now - timedelta(minutes=random.randint(10, 60))
        sched_arr = sched_dep + timedelta(hours=random.randint(2, 6))
        
        delay = random.choice([0, 0, 0, 5, 10, 15])
        status = "On Time" if delay == 0 else "Delayed"
        
        progress = random.randint(10, 90)
        
        rows.append({
            "train_id": tid,
            "train_name": name,
            "type": ttype,
            "origin": origin,
            "origin_name": origin_info["name"],
            "destination": dest,
            "dest_name": dest_info["name"],
            "scheduled_dep": sched_dep.strftime("%H:%M"),
            "scheduled_arr": sched_arr.strftime("%H:%M"),
            "delay_min": delay,
            "status": status,
            "speed_kmph": speed,
            "progress_pct": progress,
            "zone": origin_info["zone"]
        })
    return pd.DataFrame(rows)

# --------------------------- Session state init ------------------------
if "trains_df" not in st.session_state:
    st.session_state.trains_df = sample_trains()
if "incidents" not in st.session_state:
    st.session_state.incidents = []
if "audit" not in st.session_state:
    st.session_state.audit = []

# --------------------------- Layout header ----------------------------
st.markdown('<h1 class="main-header">🚆 RailOps — Indian Railways Control Dashboard</h1>', unsafe_allow_html=True)
st.write("Real-time monitoring and management of Indian Railways network")

# --------------------------- Top KPIs --------------------------------
trdf = st.session_state.trains_df

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Trains", len(trdf))
with col2:
    on_time_pct = round((trdf['delay_min'] == 0).mean() * 100, 1)
    st.metric("On-time %", f"{on_time_pct}%")
with col3:
    avg_delay = round(trdf['delay_min'].mean(), 1)
    st.metric("Avg Delay (min)", avg_delay)
with col4:
    st.metric("Avg Speed", f"{round(trdf['speed_kmph'].mean(), 1)} km/h")

# --------------------------- Status Overview ----------------------------
st.subheader("Train Status Overview")
status_cols = st.columns(3)
with status_cols[0]:
    st.info(f"**On Time:** {(trdf['status'] == 'On Time').sum()} trains")
with status_cols[1]:
    st.warning(f"**Delayed:** {(trdf['status'] == 'Delayed').sum()} trains")
with status_cols[2]:
    st.error(f"**Halted:** {(trdf['status'] == 'Halted').sum()} trains")

# --------------------------- Train List ----------------------------
st.subheader("All Trains")
for _, train in trdf.iterrows():
    with st.expander(f"{train['train_name']} (Train No: {train['train_id']})"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Route:** {train['origin_name']} → {train['dest_name']}")
            st.write(f"**Type:** {train['type']}")
            st.write(f"**Scheduled:** {train['scheduled_dep']} - {train['scheduled_arr']}")
        with col2:
            st.write(f"**Status:** {train['status']}")
            st.write(f"**Delay:** {train['delay_min']} minutes")
            st.write(f"**Progress:** {train['progress_pct']}%")
            st.write(f"**Speed:** {train['speed_kmph']} km/h")

# --------------------------- Control Center ----------------------------
st.subheader("Train Control Center")

train_options = [f"{row['train_name']} ({row['train_id']})" for _, row in trdf.iterrows()]
selected_train = st.selectbox("Select Train", options=train_options)

if selected_train:
    train_id = int(selected_train.split('(')[-1].replace(')', ''))
    train_data = trdf[trdf['train_id'] == train_id].iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Train:** {train_data['train_name']}")
        st.info(f"**Current Status:** {train_data['status']}")
        st.info(f"**Delay:** {train_data['delay_min']} minutes")
    
    with col2:
        action = st.selectbox("Action", ["Proceed as scheduled", "Hold at next station", "Change speed", "Reroute"])
        details = ""
        if action == "Hold at next station":
            details = st.slider("Hold duration (minutes)", 5, 120, 15)
        elif action == "Change speed":
            details = st.slider("New speed (km/h)", 0, 150, train_data['speed_kmph'])
        
        if st.button("Apply Action"):
            st.success(f"Action '{action}' applied to {train_data['train_name']}")

# --------------------------- Incident Management ----------------------------
st.subheader("Incident Management")

with st.form("incident_form"):
    incident_type = st.selectbox("Incident Type", 
                                ["Weather", "Technical Failure", "Track Obstruction", 
                                 "Signaling Issue", "Crowd Management", "Other"])
    incident_desc = st.text_area("Incident Description")
    severity = st.select_slider("Severity Level", options=["Low", "Medium", "High", "Critical"])
    
    if st.form_submit_button("Report Incident"):
        st.success("Incident reported successfully!")

# --------------------------- Sidebar ----------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/7/75/Indian_Railways.svg/1200px-Indian_Railways.svg.png", 
             width=80)
    st.title("RailOps Control")
    
    if st.button("🔄 Refresh Data"):
        st.session_state.trains_df = sample_trains()
        st.rerun()
    
    st.info(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    st.info(f"Active trains: {len(trdf)}")
    
    st.markdown("---")
    st.markdown("**System Status**")
    st.success("🟢 All systems operational")
    st.warning("⚠️ 2 trains delayed")
    
    st.markdown("---")
    st.markdown("**Quick Actions**")
    if st.button("Generate Daily Report"):
        st.sidebar.info("Report generation started")

st.success("Indian Railways Dashboard loaded successfully! 🚆")