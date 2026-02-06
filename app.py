import streamlit as st
import pandas as pd
import networkx as nx
import pydeck as pdk
import plotly.express as px
import joblib
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math

# -----------------------
# Page config & small css
# -----------------------
st.set_page_config(page_title="Railway ML Dashboard", layout="wide")
st.markdown("""
    <style>
      .stApp { background-color: #FAFBFF; }
      .block-container { padding-top: 0.6rem; }
      h1, h2, h3 { color: #0b3d91; }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Helpers: robust column detection
# -----------------------
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_data(csv_path="india_routes_combined.csv"):
    df = pd.read_csv(csv_path)

    # Detect lat/lon column names (robust)
    lat_src_col = pick_col(df, ["lat_source", "source_lat", "lat_src"])
    lon_src_col = pick_col(df, ["lon_source", "source_lon", "lon_src"])
    lat_dst_col = pick_col(df, ["lat_dest", "lat_destination", "destination_lat", "lat_dest"])
    lon_dst_col = pick_col(df, ["lon_dest", "lon_destination", "destination_lon", "lon_dest"])

    # If fallback missing, try other alternatives (best-effort)
    if lat_src_col is None or lon_src_col is None:
        # try columns named 'lat'/'lon' (less likely)
        lat_src_col = pick_col(df, ["lat", "latitude"]) or lat_src_col
        lon_src_col = pick_col(df, ["lon", "longitude"]) or lon_src_col

    if lat_dst_col is None or lon_dst_col is None:
        lat_dst_col = pick_col(df, ["lat_destination", "lat_dest"]) or lat_dst_col
        lon_dst_col = pick_col(df, ["lon_destination", "lon_dest"]) or lon_dst_col

    # attach chosen column names to df for later use
    df._lat_src_col = lat_src_col
    df._lon_src_col = lon_src_col
    df._lat_dst_col = lat_dst_col
    df._lon_dst_col = lon_dst_col

    return df

df = load_data()

# -----------------------
# Optional model loader (safe)
# -----------------------
@st.cache_resource
def load_model(path="railway_map_system.pkl"):
    try:
        return joblib.load(path)
    except Exception:
        return None

model = load_model()  # if not present, None

# -----------------------
# Haversine
# -----------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    a = min(1.0, max(0.0, a))
    return 2 * R * math.asin(math.sqrt(a))

# -----------------------
# Build graph (internal IDs kept)
# -----------------------
@st.cache_resource
def build_graph(df, k=8):
    G = nx.DiGraph()

    lat_src = df._lat_src_col
    lon_src = df._lon_src_col
    lat_dst = df._lat_dst_col
    lon_dst = df._lon_dst_col

    # create stations dataframe using internal node ids (name_source/name_dest)
    left = pd.DataFrame({
        "node_id": df["name_source"],
        "lat": df[lat_src] if lat_src in df.columns else np.nan,
        "lon": df[lon_src] if lon_src in df.columns else np.nan
    })
    right = pd.DataFrame({
        "node_id": df["name_dest"],
        "lat": df[lat_dst] if lat_dst in df.columns else np.nan,
        "lon": df[lon_dst] if lon_dst in df.columns else np.nan
    })
    stations = pd.concat([left, right], ignore_index=True).dropna(subset=["node_id"]).drop_duplicates("node_id").reset_index(drop=True)

    # add nodes with lat/lon (if missing lat/lon, set to 0.0 but these will appear weird on map)
    for _, r in stations.iterrows():
        lat = float(r["lat"]) if not pd.isna(r["lat"]) else 0.0
        lon = float(r["lon"]) if not pd.isna(r["lon"]) else 0.0
        G.add_node(r["node_id"], lat=lat, lon=lon)

    # add edges from dataset (use internal ids)
    for _, r in df.iterrows():
        u = r.get("name_source")
        v = r.get("name_dest")
        if pd.isna(u) or pd.isna(v):
            continue
        # safe extraction of numeric fields
        dist = float(r.get("distance_km", 0.0)) if not pd.isna(r.get("distance_km", 0.0)) else 0.0
        t = float(r.get("travel_time_hr", 0.0)) if not pd.isna(r.get("travel_time_hr", 0.0)) else (dist/60.0 if dist>0 else 0.0)
        delay = float(r.get("delay_probability", 0.0)) if not pd.isna(r.get("delay_probability", 0.0)) else 0.0
        G.add_edge(u, v, distance=dist, time=t, delay=delay, train_type=r.get("train_type", "Real"))

    # synthetic connections using k-NN among station coordinates
    coords = stations[["lat", "lon"]].values
    n_points = coords.shape[0]
    if n_points >= 2:
        n_neighbors = min(k + 1, n_points)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        for i, r in stations.iterrows():
            u = r["node_id"]
            for j in indices[i][1:]:  # skip itself
                v = stations.iloc[j]["node_id"]
                lat_u, lon_u = r["lat"], r["lon"]
                lat_v, lon_v = stations.iloc[j]["lat"], stations.iloc[j]["lon"]
                # compute haversine only if lat/lon nonzero
                try:
                    dist = haversine(float(lat_u), float(lon_u), float(lat_v), float(lon_v))
                except Exception:
                    dist = 0.0
                time = dist/60.0 if dist>0 else 0.0
                if not G.has_edge(u, v):
                    G.add_edge(u, v, distance=dist, time=time, delay=0.05, train_type="Synthetic")

    return G

G = build_graph(df)
stations_internal = sorted(G.nodes())

# -----------------------
# Mappings internal_id <-> real name
# -----------------------
# build mapping id -> real station name (prefer 'source' and 'destination' columns)
id_to_real = {}
# For nodes that appear as source in rows, map to that row's 'source' (real name)
for _, r in df.iterrows():
    ns = r.get("name_source")
    nd = r.get("name_dest")
    real_s = r.get("source")
    real_d = r.get("destination")
    if pd.notna(ns) and pd.notna(real_s):
        id_to_real.setdefault(ns, real_s)
    if pd.notna(nd) and pd.notna(real_d):
        id_to_real.setdefault(nd, real_d)

# build reverse mapping: real -> list(internal ids)
real_to_nodes_src = df.groupby("source")["name_source"].apply(lambda s: s.dropna().unique().tolist()).to_dict()
real_to_nodes_dst = df.groupby("destination")["name_dest"].apply(lambda s: s.dropna().unique().tolist()).to_dict()
# unify both source/destination mappings into one helper
def get_internal_candidates_for_real(real_name):
    cand = []
    cand.extend(real_to_nodes_src.get(real_name, []))
    cand.extend(real_to_nodes_dst.get(real_name, []))
    return list(dict.fromkeys(cand))  # unique preserve order

# -----------------------
# Delay prediction helpers
# -----------------------
def predict_delay(edge_attrs):
    # fallback: if model present, use it; else return stored delay attr
    if model is None:
        return edge_attrs.get("delay", 0.0)
    X = pd.DataFrame([{
        "distance": edge_attrs.get("distance", 0.0),
        "time": edge_attrs.get("time", 0.0),
        "train_type": edge_attrs.get("train_type", "Unknown")
    }])
    try:
        return float(model.predict_proba(X)[0][1])
    except Exception:
        try:
            return float(model.predict(X)[0])
        except Exception:
            return edge_attrs.get("delay", 0.0)

def compute_path_delay(G, path):
    if not path or len(path) < 2:
        return 0.0
    vals = []
    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            vals.append(predict_delay(G[u][v]))
    return float(np.mean(vals)) if vals else 0.0

# -----------------------
# Utility: pick best internal node among candidates
# -----------------------
def pick_best_node(candidates):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # pick the node with highest degree (most connected)
    try:
        return max(candidates, key=lambda n: G.degree(n))
    except Exception:
        return candidates[0]

# -----------------------
# Map plotting
# -----------------------
def plot_path(G, path=None, highlight_nodes=None):
    # highlight_nodes: list of node ids to draw as big markers
    all_nodes = [
        {"lat": float(G.nodes[n].get("lat", 0.0)),
         "lon": float(G.nodes[n].get("lon", 0.0)),
         "name": id_to_real.get(n, str(n))}
        for n in G.nodes()
    ]

    layers = []
    node_layer_all = pdk.Layer(
        "ScatterplotLayer",
        all_nodes,
        get_position=["lon", "lat"],
        get_color=[90, 200, 150],
        get_radius=9000,
        pickable=True,
        auto_highlight=True
    )
    layers.append(node_layer_all)

    if path and len(path) >= 2:
        edges = []
        nodes_path = []
        for u, v in zip(path[:-1], path[1:]):
            edges.append({
                "from_lat": float(G.nodes[u].get("lat", 0.0)),
                "from_lon": float(G.nodes[u].get("lon", 0.0)),
                "to_lat": float(G.nodes[v].get("lat", 0.0)),
                "to_lon": float(G.nodes[v].get("lon", 0.0))
            })
        edge_layer = pdk.Layer(
            "LineLayer",
            edges,
            get_source_position=["from_lon", "from_lat"],
            get_target_position=["to_lon", "to_lat"],
            get_color=[200, 60, 30],
            get_width=4
        )
        layers.append(edge_layer)

        nodes_path = [
            {"lat": float(G.nodes[n].get("lat", 0.0)),
             "lon": float(G.nodes[n].get("lon", 0.0)),
             "name": id_to_real.get(n, str(n))}
            for n in path
        ]
        node_layer_path = pdk.Layer(
            "ScatterplotLayer",
            nodes_path,
            get_position=["lon", "lat"],
            get_color=[0, 90, 200],
            get_radius=18000,
            pickable=True
        )
        layers.append(node_layer_path)
    elif highlight_nodes:
        nodes_path = [
            {"lat": float(G.nodes[n].get("lat", 0.0)),
             "lon": float(G.nodes[n].get("lon", 0.0)),
             "name": id_to_real.get(n, str(n))}
            for n in highlight_nodes if n in G.nodes()
        ]
        if nodes_path:
            node_layer_path = pdk.Layer(
                "ScatterplotLayer",
                nodes_path,
                get_position=["lon", "lat"],
                get_color=[255, 80, 80],
                get_radius=22000,
                pickable=True
            )
            layers.append(node_layer_path)

    view_state = pdk.ViewState(latitude=22.97, longitude=78.65, zoom=4.5, pitch=0)
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    tooltip={"text": "{name}"})
    return deck

# -----------------------
# UI Navigation
# -----------------------
st.sidebar.title("🚉 Navigation")
page = st.sidebar.radio("Go to", ["Route Finder", "Delay Predictor", "Analytics", "Station Explorer"])

# -----------------------
# Route Finder (real names shown; graph uses internal IDs)
# -----------------------
if page == "Route Finder":
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("📍 Route Finder")

    # show sidebar controls only on this page (keeps UI clean on other pages)
    st.sidebar.subheader("Route Finder Controls")
    src_real = st.sidebar.selectbox("Source (real name)", options=sorted(df["source"].dropna().unique()))
    dst_real = st.sidebar.selectbox("Destination (real name)", options=sorted(df["destination"].dropna().unique()))
    find_pressed = st.sidebar.button("Find Shortest Path")

    # map real -> internal (candidates)
    src_cands = get_internal_candidates_for_real(src_real)
    dst_cands = get_internal_candidates_for_real(dst_real)
    src_node = pick_best_node(src_cands)
    dst_node = pick_best_node(dst_cands)

    if find_pressed:
        if src_node is None or dst_node is None:
            st.error("Could not map selected station(s) to internal graph nodes. Try another pair.")
        else:
            try:
                path = nx.shortest_path(G, source=src_node, target=dst_node, weight="time")
            except nx.NetworkXNoPath:
                st.error("No path found between the selected stations.")
            except Exception as e:
                st.error(f"Error finding path: {e}")
            else:
                # present path in real readable names
                path_readable = [id_to_real.get(n, str(n)) for n in path]
                st.success(" → ".join(path_readable))

                # compute simple route stats
                total_dist = 0.0
                total_time = 0.0
                for u, v in zip(path[:-1], path[1:]):
                    if G.has_edge(u, v):
                        total_dist += float(G[u][v].get("distance", 0.0))
                        total_time += float(G[u][v].get("time", 0.0))
                col1, col2, col3 = st.columns(3)
                col1.metric("Stops", len(path))
                col2.metric("Total Distance (km)", f"{total_dist:.1f}")
                col3.metric("Estimated Time (hr)", f"{total_time:.2f}")

                # show map with route highlighted
                st.pydeck_chart(plot_path(G, path), use_container_width=True)
    else:
        # show full network map by default
        st.info("Select source & destination from the sidebar and click 'Find Shortest Path'")
        st.pydeck_chart(plot_path(G), use_container_width=True)

# -----------------------
# Delay Predictor (real names in UI)
# -----------------------
elif page == "Delay Predictor":
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("🚦 Delay Predictor")
    src_real = st.selectbox("Source (real name)", options=sorted(df["source"].dropna().unique()))
    dst_real = st.selectbox("Destination (real name)", options=sorted(df["destination"].dropna().unique()))
    predict_pressed = st.button("Predict Delays")

    src_node = pick_best_node(get_internal_candidates_for_real(src_real))
    dst_node = pick_best_node(get_internal_candidates_for_real(dst_real))

    if predict_pressed:
        if src_node is None or dst_node is None:
            st.error("Mapping to internal nodes failed for selected station(s).")
        else:
            try:
                path = nx.shortest_path(G, source=src_node, target=dst_node, weight="time")
            except nx.NetworkXNoPath:
                st.error("No path found between selected stations.")
            except Exception as e:
                st.error(f"Error computing path: {e}")
            else:
                # show readable path
                path_readable = [id_to_real.get(n, str(n)) for n in path]
                delay_avg = compute_path_delay(G, path)
                st.subheader("Primary Path")
                st.success(f"{' → '.join(path_readable)}")
                st.metric("Average Predicted Delay", f"{delay_avg:.3f}")
                st.pydeck_chart(plot_path(G, path), use_container_width=True)

                # attempt alternate path if beneficial
                try:
                    from networkx.algorithms.simple_paths import shortest_simple_paths
                    k_paths = shortest_simple_paths(G, src_node, dst_node, weight="time")
                    alt = None
                    # check a few next candidates
                    count = 0
                    for p in k_paths:
                        if count == 0:
                            count += 1
                            continue  # first is primary
                        if count > 5:
                            break
                        p_delay = compute_path_delay(G, list(p))
                        if p_delay < delay_avg - 1e-6:
                            alt = list(p)
                            break
                        count += 1
                    if alt:
                        alt_read = [id_to_real.get(n, str(n)) for n in alt]
                        alt_delay = compute_path_delay(G, alt)
                        st.subheader("Alternative Path (Better Delay)")
                        st.warning(f"{' → '.join(alt_read)} (Avg Delay {alt_delay:.3f})")
                        st.pydeck_chart(plot_path(G, alt), use_container_width=True)
                except Exception:
                    # ignore if simple_paths can't run or is expensive
                    pass

# -----------------------
# Analytics
# -----------------------
elif page == "Analytics":
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("📊 Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Avg Delay by Train Type")
        fig = px.histogram(df, x="train_type", y="delay_probability", histfunc="avg", color="train_type")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Top 10 Busiest Stations (real names)")
        # show busiest by real source names
        busiest = df["source"].value_counts().head(10).reset_index()
        busiest.columns = ["Station", "Connections"]
        fig2 = px.bar(busiest, x="Station", y="Connections", color="Connections")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Compare Metric by Train Type")
    metric = st.selectbox("Metric", ["avg_speed_kmph", "distance_km", "ticket_price_inr", "travel_time_hr"])
    metric_df = df.groupby("train_type")[metric].mean().reset_index()
    fig3 = px.pie(metric_df, names="train_type", values=metric, hole=0.3)
    fig3.update_traces(textinfo="label+percent+value", texttemplate="%{label}: %{value:.1f} (%{percent})")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Station Map")
    st.pydeck_chart(plot_path(G), use_container_width=True)

# -----------------------
# Station Explorer
# -----------------------
elif page == "Station Explorer":
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("🔍 Station Explorer")
    all_reals = sorted(pd.concat([df["source"].dropna(), df["destination"].dropna()]).unique())
    stn = st.selectbox("Select station (real name)", options=all_reals)
    sub_df = df[(df["source"] == stn) | (df["destination"] == stn)]
    st.dataframe(sub_df.reset_index(drop=True), use_container_width=True)

    # highlight a representative internal node on the map
    candidates = get_internal_candidates_for_real(stn)
    highlight = []
    if candidates:
        # choose best-connected one for highlight
        highlight_node = pick_best_node(candidates)
        highlight = [highlight_node]
    st.pydeck_chart(plot_path(G, highlight_nodes=highlight), use_container_width=True)
