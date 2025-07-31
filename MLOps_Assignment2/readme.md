
# 🧪 ML Experiment Tracking with MLflow and Weights & Biases

This project demonstrates how to track machine learning experiments using **MLflow** and **Weights & Biases (W&B)**. It trains a classifier on the Iris dataset and logs parameters, metrics, models, and artifacts for comparison and reproducibility.

---

## 📁 Project Structure

```
MLOps_assignments2/
├── mlflow_tracking.py         # MLflow logging script
├── wandb_tracking.py          # Weights & Biases logging script
├── common_code.py             # Shared functions (load, train, evaluate)
├── outputs/
│   └── confusion_matrix.png   # Logged artifact (generated plot)
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Setup Instructions

### 🔹 Clone the Repository
```bash
git clone https://github.com/your-username/experiment-tracking.git
cd experiment-tracking
```

### 🔹 Create and Activate a Virtual Environment
```bash
python -m venv tracking-env
# Activate
# On Windows:
tracking-env\Scripts\activate
# On macOS/Linux:
source tracking-env/bin/activate
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```
Or install manually:

```bash
pip install mlflow wandb scikit-learn pandas matplotlib seaborn
```

---

## 📊 Track Experiments with MLflow

### 🔸 Run the MLflow Script
```bash
python mlflow_tracking.py
```

### 🔸 Launch the MLflow UI
```bash
mlflow ui
```
Then visit: [http://localhost:5000](http://localhost:5000)

### 🔸 MLflow Logs:
- Parameters: `n_estimators`, `max_depth`
- Metrics: `accuracy`, `precision`, `recall`
- Artifacts: Model and confusion matrix image

---

## 📈 Track Experiments with Weights & Biases (W&B)

### 🔸 Login to W&B
```bash
wandb login
```
This opens your browser and connects to your W&B account.

### 🔸 Run the W&B Script
```bash
python wandb_tracking.py
```

### 🔸 W&B Logs:
- Parameters and metrics
- Confusion matrix image as artifact
- Auto-generated experiment dashboard

Visit: [https://wandb.ai/](https://wandb.ai/) → Your Project: `iris-tracking`

---

## 📌 Feature Comparison: MLflow vs W&B

| Feature                | MLflow UI       | Weights & Biases    |
|------------------------|-----------------|----------------------|
| Interface              | Local Web UI    | Cloud Dashboard      |
| Metric Logging         | ✅               | ✅                   |
| Model Logging          | ✅               | ✅                   |
| Artifact Logging       | ✅ Manual        | ✅ Automatic/Easy    |
| Collaboration          | ❌ Local only    | ✅ Team sharing      |
| Hyperparameter Sweeps  | 🟡 Limited       | ✅ Built-in          |
| Visualizations         | ✅ Static charts | ✅ Interactive plots |
| Auto-Logging           | ✅ mlflow.sklearn.autolog() | ✅ wandb.watch() |

---

## 📎 Files Overview

- ✅ `mlflow_tracking.py` – Tracks experiments with MLflow using manual logging for metrics, params, models, and artifacts.
- ✅ `wandb_tracking.py` – Tracks experiments with Weights & Biases using their cloud dashboard.
- ✅ `common_code.py` – Shared functions for loading data, training models, evaluating, and plotting confusion matrix.

---

## 🎯 Bonus Ideas

- Use `mlflow.sklearn.autolog()` or `wandb.watch()` to simplify logging.
- Try different classifiers and compare results.
- Use W&B Sweeps for automatic hyperparameter optimization.
- Push models to W&B Artifacts or MLflow Model Registry.
- Build a Streamlit dashboard to show experiment results dynamically.

---

## 🧑‍💻 Author

**Mrunal Talikoti**  
3rd Year CSE-AI | AI/ML & GenAI Enthusiast  
🔗 [LinkedIn](#) • 💻 [GitHub](#)

---

## 📜 License

This project is licensed under the MIT License.
