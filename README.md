# NIFTY 50 Financial Machine Learning & MLOps Portfolio
### End-to-End Modeling, Experiment Tracking, Explainability, Containerization, CI/CD & Dashboard

**Author & Quantitative Developer:** Manasa Premnathan  
**GitHub Profile:** [https://github.com/manasa-premnathan05](https://github.com/manasa-premnathan05)  
**Public Repository:** [https://github.com/manasa-premnathan05/nifty50-mlops-portfolio](https://github.com/manasa-premnathan05/nifty50-mlops-portfolio)  
**Interactive Streamlit Portal:** [https://manasa-premnathan05-nifty50-mlops.streamlit.app](https://manasa-premnathan05-nifty50-mlops.streamlit.app)  

---

## 🏛️ System Architecture

```
                                  +---------------------------------------+
                                  | Raw Merged Dataset (60,550 records)   |
                                  +-------------------+-------------------+
                                                      |
                                           [Data Hygiene & Curation]
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Cleaned NIFTY 50 Dataset (29,310 rows)|
                                  +-------------------+-------------------+
                                                      |
                          +---------------------------+---------------------------+
                          |                                                       |
                          v                                                       v
               +--------------------+                                  +--------------------+
               | Exp 4: ML Modeling |                                  | Exp 5: XAI &       |
               | & MLflow Tracking  |                                  | Fairness (Fairlearn|
               +----------+---------+                                  +----------+---------+
                          |                                                       |
                          +---------------------------+---------------------------+
                                                      |
                                                      v
                                           +---------------------+
                                           | best_model.pkl      |
                                           | (Decision Tree)     |
                                           +----------+----------+
                                                      |
                   +----------------------------------+----------------------------------+
                   |                                  |                                  |
                   v                                  v                                  v
         +-------------------+              +-------------------+              +-------------------+
         | Exp 6: FastAPI    |              | Exp 7: GitHub     |              | Exp 8: Streamlit  |
         | REST Service &    |              | Actions CI/CD     |              | Dashboard & RAI   |
         | Docker Container  |              | Automated Testing |              | Governance Report |
         +-------------------+              +-------------------+              +-------------------+
```

---

## 📊 Dataset Overview: Raw vs. Cleaned Data Pipeline

| Dataset Attribute | Raw Merged Dataset (`raw_merged_dataset.csv`) | Cleaned Feature Dataset (`final_nifty50_dataset.csv`) |
| :--- | :--- | :--- |
| **Total Observations** | 60,550 historical rows | 29,310 curated trading sessions |
| **Asset Universe** | Unfiltered equities across diverse categories | 42 official NIFTY 50 constituent stocks |
| **Feature Width** | 11 raw market attributes (OHLCV, Market Cap) | 24 columns (16 engineered technical features) |
| **Temporal Alignment** | Gaps, non-synchronous holidays, duplicates | Strictly ordered 2018–2021 chronological sessions |
| **Engineered Signals** | None (pure raw quotes) | Moving Averages (20/50), RSI, MACD, Lagged Returns |

---

## 📁 Repository Structure

```
├── .github/
│   └── workflows/
│       └── ci_cd.yml                     # Multi-stage GitHub Actions CI/CD workflow
├── tests/
│   ├── test_model_inference.py           # Pytest unit tests for model contracts
│   └── test_api_endpoints.py             # Pytest integration tests for FastAPI
├── app.py                                # Production FastAPI REST service
├── dashboard.py                          # Streamlit interactive quantitative dashboard
├── Dockerfile                            # Multi-stage, non-root production Dockerfile
├── .dockerignore                         # Container context exclusion rules
├── requirements.txt                      # Pinned Python dependencies
├── best_model.pkl                        # Serialized production model artifact
├── dvc.yaml                              # Data Version Control stage manifest
├── Responsible_AI.md                     # Comprehensive Responsible AI governance report
├── final_nifty50_dataset.csv             # Curated dataset (29,310 rows, 16 features)
├── final_nifty50_dataset.xlsx            # Excel version of curated dataset
├── raw_merged_dataset.csv                # Raw merged historical dataset (60,550 rows)
├── Experiment_4_Modeling_Tracking.ipynb   # Experiment 4 executable Colab notebook
├── Experiment_5_XAI_Fairness.ipynb       # Experiment 5 executable Colab notebook
├── Experiment_6_Containerization_API.ipynb # Experiment 6 executable Colab notebook
├── Experiment_7_CICD_Pipeline.ipynb      # Experiment 7 executable Colab notebook
└── Experiment_8_Dashboard_Portfolio.ipynb # Experiment 8 executable Colab notebook
```

---

## 🚀 How to Push this Repository to GitHub

To push this complete project to your GitHub account:

```bash
# 1. Initialize local git repository
git init

# 2. Add all project files
git add .

# 3. Commit with a formal release message
git commit -m "Initial Release: NIFTY 50 Quantitative ML, CI/CD, FastAPI & Responsible AI Portfolio"

# 4. Set main branch
git branch -M main

# 5. Connect to your GitHub repository
git remote add origin https://github.com/manasa-premnathan05/nifty50-mlops-portfolio.git

# 6. Push all commits to GitHub
git push -u origin main
```

---

## ⚡ Local Quickstart Commands

### 1. Launch FastAPI REST Service
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger Docs: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`
- Directional Prediction: `POST http://localhost:8000/predict`

### 2. Run Automated Test Suite (Pytest)
```bash
pytest tests/ -v
```

### 3. Build & Run Docker Container
```bash
docker build -t nifty50-predictor:v1 .
docker run -d -p 8000:8000 --name nifty50-api nifty50-predictor:v1
```

### 4. Run Interactive Quantitative Dashboard (Streamlit)
```bash
streamlit run dashboard.py
```
"""

with open(os.path.join(BASE_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)
print(f"Created {os.path.join(BASE_DIR, 'README.md')}")
