# Advanced Predictive Maintenance System (Full Stack)

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-FF7F0E?style=for-the-badge&logo=XGBoost)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

A production-grade end-to-end machine learning pipeline designed to predict the **Remaining Useful Life (RUL)** of aircraft engines using the NASA C-MAPSS dataset.

This project demonstrates a **Full Stack Data Engineering Architecture** separating Training (Python Scripts), Inference (FastAPI Backend), and Visualization (Streamlit Frontend).

</div>

---

## Key Features

- **Hybrid Detection System:** Combines Supervised Learning (**XGBoost** for Regression) with Unsupervised Learning (**Isolation Forest** for Anomaly Detection) for robust safety monitoring.
- **Automated Optimization:** Integrated **Optuna** for hyperparameter tuning to maximize model performance.
- **Explainability (XAI):** Implemented **SHAP (SHapley Additive exPlanations)** to provide actionable insights into individual predictions.
- **Modular Architecture:** Decoupled Backend (FastAPI) and Frontend (Streamlit) following industry standard microservices patterns.
- **Smart Data Generation:** Includes "Smart Randomizer" in frontend that loads realistic data to demonstrate system behavior without triggering false anomalies.
- **Production Ready:** Includes Docker support for containerized deployment.

---

## System Architecture

The system follows a scalable Client-Server architecture designed for real-world deployment.

```text
+-------------------+      HTTP POST      +-------------------+
| Streamlit App     | -----------------> | FastAPI Backend   |
| (Visualization)   | <----------------- | (Inference)       |
+-------------------+      JSON Response  +-------------------+
        |                                       |
        |                                       v
        |                              +-------------------+
        |                              | XGBoost Model     |
        |                              | (.pkl Artifact)   |
        |                              +-------------------+
        |
        v
+-------------------+
| Static Assets     |
| (Metrics/Graphs)  |
+-------------------+
```

---

## Project Structure

```text
predictive-maintenance-demo/
│
├── train_advanced.py       # Backend: Trains Model + Optimizer + Metrics
├── api.py                  # Backend: FastAPI REST Endpoint
├── app.py                  # Frontend: Streamlit Dashboard
├── requirements.txt        # Python Dependencies (Python 3.12 / M3 Compatible)
│
├── src/
│   ├── __init__.py         # Package Marker
│   ├── pipeline.py         # Logic: Preprocessing, FE, Hybrid Inference
│   └── utils.py            # Logic: Visualization (Graphs)
│
├── models/                 # Saved Artifacts
│   ├── advanced_model.pkl  # XGBoost + Scaler + IsolationForest
│   └── metrics.json        # Performance Metrics (RMSE, R2)
│
├── img/                    # Generated Analytics Graphs
│   ├── feature_importance.png
│   └── residual_plot.png
│
└── data/
    └── raw/                # Place NASA Data here
        ├── train_FD001.txt
        ├── test_FD001.txt
        └── RUL_FD001.txt
```

---

## Getting Started

### Prerequisites

- **Python 3.8+** (Tested on Python 3.12).
- **NASA C-MAPSS Dataset (FD001)**. Download from [Official NASA Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

### 1. Installation

Clone the repository and install dependencies.

```bash
git clone https://github.com/shhubham311/predictive-maintenance-demo.git
cd predictive-maintenance-demo

# Create Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Libraries (Optimized for Mac M3 / Python 3.12)
pip install -r requirements.txt
```

### 2. Data Setup

Download the FD001 subset files and place them in `data/raw/`:
1. `train_FD001.txt`
2. `test_FD001.txt`
3. `RUL_FD001.txt`

### 3. Training Model

Run the training script. This performs **Optimization**, trains the **Hybrid System**, and generates **Visualizations**.

```bash
python train_advanced.py
```

*Output:* Creates `models/advanced_model.pkl`, `metrics.json`, and PNG files in `img/`.

### 4. Running Application

You need **two terminals** to run the full architecture (Backend + Frontend).

**Terminal 1 (Backend API):**
```bash
uvicorn api:app --reload
```
*Server runs at `http://localhost:8000`*

**Terminal 2 (Frontend Dashboard):**
```bash
streamlit run app.py
```
*Dashboard runs at `http://localhost:8501`*

---

## Model Performance

The model is trained on the NASA C-MAPSS FD001 subset using XGBoost with engineered features.

| Metric | Score | Description |
|--------|-------|-------------|
| Validation RMSE | 36.80 | Root Mean Squared Error (Average error in cycles) |
| Validation R² | 0.7036 | Coefficient of Determination (Explains ~70% of variance) |

*(Metrics are generated dynamically and saved in `models/metrics.json`)*

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.12 |
| **ML Framework** | XGBoost, Scikit-Learn |
| **Deep Learning** | TensorFlow (Optional - commented out) |
| **Optimization** | Optuna |
| **Explainability** | SHAP |
| **API** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Containerization** | Docker |

---

## Dashboard Guide

The Streamlit dashboard features three main modes for demos:

1. **Smart Random (Auto-Fill):**
   - **Healthy Data:** Loads a brand new engine snapshot.
   - **Real Data:** Picks a random row from NASA training data to show model variance.
   - **Simulate Failing:** Loads a "worn-out" sensor snapshot to trigger Critical status.
2. **Manual Input:** Allows specific sensor tuning for edge case testing.
3. **Explainability (SHAP):** Generates a force plot explaining which sensors pushed the RUL prediction up or down.

---

## Author

**Shubham Kumar**  
*Data Engineering & AI/ML Intern*

- [LinkedIn](https://www.linkedin.com/in/shubhamkumar311/)
- [GitHub](https://github.com/shhubham311)
- [Email](mailto:shubham31103@gmail.com)

---

## License

This project is for educational and portfolio purposes.