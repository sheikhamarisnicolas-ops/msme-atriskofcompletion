# 📊 MSME Project Completion Predictor
### DOST SETUP 4.0 iFund Program — Western Visayas
**Model:** XGBoost (Extreme Gradient Boosting)
**Author:** Shi

---

## 📌 Overview

This project predicts whether an MSME (Micro, Small, and Medium Enterprise) beneficiary will **complete** their technology project under the DOST SETUP 4.0 iFund Program in Western Visayas.

It uses **XGBoost**, a gradient boosting machine learning model, trained on 321 MSME records across 6 provinces (Aklan, Antique, Capiz, Guimaras, Iloilo, Negros Occidental).

---

## 🎯 Target Variable

| Value | Meaning |
|---|---|
| 1 | Completed |
| 0 | Not Completed |

---

## 📂 Project Structure

```
MSME_XGBoost/
├── MSME_data_cleaned.xlsx          ← Dataset (321 records, 6 provinces)
├── SHI_XGBoost_Final.py            ← Python script (trains model, saves pkl)
├── SHI_XGBoost_Streamlit_Final.py  ← Streamlit web app (prediction tool)
├── SHI_XGBoost_Final.ipynb         ← Jupyter notebook (full documentation)
├── requirements.txt                ← Python dependencies
└── README.md                       ← This file
```

> **Note:** `xgb_final.pkl` is not included in the repo.
> Run `SHI_XGBoost_Final.py` to generate it.

---

## ⚙️ Process Flow

Following CRISP-DM framework and adviser's guidelines:

```
Raw Data (321 records)
  → Drop leakage columns (Refund_Status, Beneficiary_Name, Year)
  → Train-Test Split 80/20 — stratified — BEFORE any cleaning
  → Normalize & Encode TRAIN and TEST separately (no leakage)
  → Feature Engineering (Cost_to_Size_Ratio, Log_Project_Cost)
  → SMOTE on TRAIN only — fix class imbalance
  → RandomizedSearchCV — auto hyperparameter tuning (optimize AUC)
  → Evaluate on TEST SET only (AUC, F1, Confusion Matrix)
  → Save model as xgb_final.pkl
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Cross-Validation AUC (5-fold) | 0.8422 |
| Test AUC | 0.7289 |
| Weighted F1-Score | 0.6874 |
| F1 (Not Completed) | 0.5532 |
| F1 (Completed) | 0.7470 |

---

## 🔑 Key Design Decisions (per Adviser Sir Paolo)

- ✅ `Refund_Status` dropped **before** splitting — prevents data leakage
- ✅ **Split first, clean after** — train and test cleaned separately
- ✅ LabelEncoders fit on **training set only**, applied to test
- ✅ **SMOTE on training set only** — never on test set
- ✅ Hyperparameters **auto-tuned** via RandomizedSearchCV — not fixed
- ✅ Primary metrics: **AUC and F1-score** — not accuracy
- ✅ Confusion matrix from **test set predictions only**
- ✅ Aklan included — 17 records added from new cleaned dataset

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/MSME_XGBoost.git
cd MSME_XGBoost
```

### 2. Create and activate Anaconda environment
```bash
conda create -n msme_xgb python=3.10
conda activate msme_xgb
pip install -r requirements.txt
```

### 3. Train the model
```bash
python SHI_XGBoost_Final.py
```
This generates `xgb_final.pkl` and saves chart images.

### 4. Launch Streamlit app
```bash
streamlit run SHI_XGBoost_Streamlit_Final.py
```
Opens at `http://localhost:8501`

### 5. Or open the Jupyter notebook
```bash
jupyter notebook SHI_XGBoost_Final.ipynb
```

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
xgboost
streamlit
openpyxl
scipy
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📚 References

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.*
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.

---

## ⚠️ Notes

- Aklan was previously excluded due to data coverage gaps but is now included in `MSME_data_cleaned.xlsx`
- Model trained and evaluated following the **CRISP-DM** framework
- This is part of a group study comparing multiple ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
