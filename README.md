# Railway-ML-Analytics-System
# 🚆 Railway ML Analytics & Operations System

An end-to-end **railway analytics and decision-support platform** that combines **graph algorithms, machine learning, geospatial visualization, and interactive dashboards** to analyze railway routes, predict delays, and simulate operational control.

This project demonstrates how real-world transportation systems can be modeled as **graph-based networks** and enhanced using **ML-driven insights**.

---

## 🔍 Project Overview

The system models the railway network as a **directed graph**, where:
- **Stations** are nodes  
- **Routes** are edges enriched with distance, travel time, and delay attributes  

On top of this graph, the system enables:
- Optimal route discovery
- Delay probability estimation using ML
- Interactive map-based visualization
- Railway operations monitoring via dashboards

---

## ✨ Key Features

### 🧠 Graph-Based Route Optimization
- Railway network modeled using **NetworkX**
- Shortest-path routing based on **travel time**
- Robust handling of sparse connectivity
- Synthetic route generation using **k-Nearest Neighbors (k-NN)** on geographic coordinates

---

### 🚦 Machine Learning–Based Delay Prediction
- Predicts **average delay probability** along a route
- Uses features such as:
  - distance  
  - travel time  
  - train type  
- Supports loading trained ML models via **joblib**
- Graceful fallback when a trained model is unavailable

---

### 🗺️ Geospatial Visualization
- Interactive maps built using **PyDeck**
- Visualizes:
  - entire railway network  
  - selected routes  
  - station-level details  
- Uses **Haversine distance** for accurate geographic calculations

---

### 📊 Analytics Dashboard
- Delay analysis by train type
- Station connectivity and traffic insights
- Metric comparisons (speed, distance, travel time, ticket price)
- Interactive charts using **Plotly**

---

### 🏗️ RailOps Control Dashboard
- Simulated real-time railway operations
- KPI monitoring:
  - On-time percentage  
  - Average delay  
  - Average speed  
- Train-level control actions
- Incident reporting and severity classification

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Framework:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Graph Algorithms:** NetworkX  
- **Machine Learning:** Scikit-learn  
- **Visualization:** PyDeck, Plotly  
- **Model Persistence:** Joblib  

---
