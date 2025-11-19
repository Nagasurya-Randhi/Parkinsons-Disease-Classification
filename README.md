# Parkinson's Disease Classification

**One-line:** A machine-learning project that analyzes speech features to classify Parkinson's disease using ensemble methods (Random Forest) 

---

## 🔍 Project overview

This repository contains code, data and a mini-presentation for a Parkinson's Disease (PD) classification study based on speech features. The primary objective is to preprocess the data, explore feature relationships, train classification models (Random Forest as the primary model), and evaluate model performance using standard metrics.

**Primary files in this repo:**

* `Parkinson Classification.ipynb` — End-to-end Jupyter Notebook with EDA, preprocessing, training and evaluation.
* `pd_speech_features.csv` — Tabular dataset containing speech-derived features for PD classification.
* `minipj.pptx` — Short presentation summarizing the project and results.
* `README.md` — (This file) polished project README.

> Note: I generated this README after reviewing the repository index. If you want numbers (accuracy, confusion matrix images, final model file path), I left clear placeholders and instructions where to insert those exact values/screenshots.

---

## 🚀 Highlights / What this repo demonstrates

* Data cleaning and feature engineering for biomedical signals.
* Exploratory Data Analysis (visual + numeric) to identify predictive signals.
* Supervised classification using Random Forest (and comparison with other baselines where implemented).
* Model evaluation using accuracy, precision, recall, F1-score and confusion matrix.
* Reproducible notebook that ties the entire workflow together for teaching or prototyping.

---

## 🧭 Quickstart (run locally)

1. Clone the repo:

```bash
git clone https://github.com/Nagasurya-Randhi/Parkinson-s-disease-classification.git
cd Parkinson-s-disease-classification
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt || pip install jupyter pandas numpy scikit-learn matplotlib seaborn
```

3. Launch the notebook:

```bash
jupyter notebook "Parkinson Classification.ipynb"
```

4. Follow the notebook cells: EDA → preprocessing → model training → evaluation.

---

## 📁 Files & structure (explanation)

* **Parkinson Classification.ipynb** — The notebook should contain:

  * Dataset load and inspection
  * Missing value handling and feature scaling
  * EDA plots (feature distributions, correlations, class balance)
  * Train/test split, cross-validation
  * Model training (Random Forest primary)
  * Performance evaluation and visualization
  * (Optional) feature importance and brief interpretation

* **pd_speech_features.csv** — Raw/cleaned speech features used for training. Use `pandas.read_csv()` to load.

* **minipj.pptx** — A concise presentation summarizing objectives, dataset, approach, and conclusions. Useful when sharing results with non-technical audiences.

---

## 🛠 Suggested requirements (create `requirements.txt`)

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
ipykernel
joblib
```
## ✅ Reproducibility checklist

1. Confirm `pd_speech_features.csv` is the same version used in the notebook (file hash or timestamp).
2. Set a random seed before model training for reproducible splits: `random_state=42` (or your chosen seed).
3. If cross-validation is used, specify number of folds and scoring metric.
4. Save final model using `joblib.dump(model, "rf_model.joblib")` and include the path in this README.

---

## 🔬 Notes on methodology (suggested best-practices)

* **Scaling:** Many speech features benefit from standard scaling (`StandardScaler`) or `RobustScaler` where outliers exist.
* **Imbalance handling:** If classes are imbalanced, consider stratified splitting, class-weighted models, or resampling (SMOTE/undersampling).
* **Feature selection:** Try model-based selection (feature importances from RF) and recursive feature elimination to improve generalization.
* **Validation:** Use stratified k-fold cross-validation to get stable performance estimates.

---
## 🤝 Contributing

If you'd like contributions:

* Create issues describing desired improvements (e.g., add CI, tests, packaging, convert notebook to scripts).
* Prefer PRs that include tests (for script conversions) and update `requirements.txt`.

---

## ✉️ Contact / Maintainer

Nagasurya-Randhi (repository owner)

---

## Appendix: Short maintenance tasks (optional)

1. Add `requirements.txt` and `environment.yml` for conda users.
2. Convert key notebook sections into `src/` scripts (`data.py`, `features.py`, `train.py`, `evaluate.py`) for reproducible CLI runs.
3. Add a GitHub Action to run notebook linting or tests on PRs.
