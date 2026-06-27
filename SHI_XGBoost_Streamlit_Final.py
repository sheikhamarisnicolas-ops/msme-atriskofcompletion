"""
Streamlit App — XGBoost MSME Project Completion Predictor (Final)
DOST SETUP 4.0 iFund Program, Western Visayas

HOW TO RUN:
  1. Run SHI_XGBoost_Final.py first to generate xgb_final.pkl
  2. streamlit run SHI_XGBoost_Streamlit_Final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import ConfusionMatrixDisplay

st.set_page_config(
    page_title = "MSME Completion Predictor",
    page_icon  = "📊",
    layout     = "wide"
)

# ===========================================================================
# LOAD MODEL
# ===========================================================================
@st.cache_resource
def load_model():
    with open('xgb_final.pkl', 'rb') as f:
        return pickle.load(f)

try:
    art          = load_model()
    model        = art['model']
    encoders     = art['encoders']
    sector_map   = art['sector_map']
    size_num_map = art['size_num_map']
    feature_cols = art['feature_cols']
    best_params  = art['best_params']
    cv_auc       = art['cv_auc']
    test_auc     = art['test_auc']
    test_f1      = art['test_f1']
    f1_nc        = art['f1_nc']
    f1_c         = art['f1_c']
    importance_df= art['importance_df']
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ===========================================================================
# PREPROCESS — same logic as training script
# ===========================================================================
def preprocess_input(province, sector, ownership, size, project_cost, has_prior):
    sector = sector_map.get(sector, sector)
    if sector.startswith('Others'):
        sector = 'Others (grouped)'

    row = pd.DataFrame([{
        'Province'          : province,
        'Sector'            : sector,
        'type_of_ownership' : ownership,
        'size_of_enterprise': size,
        'Project_Cost'      : project_cost,
        'Has_Prior_Funding' : int(has_prior)
    }])

    for col in ['Province','Sector','type_of_ownership','size_of_enterprise']:
        le  = encoders[col]
        val = str(row[col].iloc[0])
        row[col] = le.transform([val])[0] if val in le.classes_ else -1

    size_num = size_num_map.get(size.lower(), 1)
    row['Cost_to_Size_Ratio'] = project_cost / size_num
    row['Log_Project_Cost']   = np.log1p(project_cost)

    return row[feature_cols]

# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.title("📊 MSME Predictor")
    st.caption("DOST SETUP 4.0 iFund\nWestern Visayas")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home",
        "🔮 Predict Completion",
        "📈 Model Performance",
        "📊 Feature Importance",
        "ℹ️ About"
    ])
    st.divider()
    if model_loaded:
        st.success("Model ready ✅")
        st.caption(f"CV AUC: {cv_auc:.4f}\nTest AUC: {test_auc:.4f}")
    else:
        st.error("Model not loaded\nRun the .py script first")

# ===========================================================================
# HOME
# ===========================================================================
if page == "🏠 Home":
    st.title("📊 MSME Project Completion Predictor")
    st.subheader("DOST SETUP 4.0 iFund Program — Western Visayas")
    st.markdown("Predicts whether an MSME beneficiary will **complete** their technology project using **XGBoost**.")

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total records",  "321")
    c2.metric("Completed",      "223  (69.5%)")
    c3.metric("Not Completed",  "98   (30.5%)")
    c4.metric("Provinces",      "6 incl. Aklan")

    st.divider()
    st.markdown("### Process Flow (per Adviser Sir Paolo)")
    st.code("""
Raw Data (321 records)
  ↓
Drop leakage columns — Refund_Status, Beneficiary_Name, Year
  ↓
Train-Test Split  80 / 20  (stratified on Completion_Status)
  ↓
Normalize & Encode  →  TRAIN and TEST separately (no leakage)
  ↓
Feature Engineering  →  Cost_to_Size_Ratio, Log_Project_Cost
  ↓
SMOTE  →  training set only  (fixes class imbalance)
  ↓
RandomizedSearchCV  →  auto-tune hyperparameters  (optimize AUC)
  ↓
Evaluate on TEST SET only  →  AUC, F1, Confusion Matrix
  ↓
Save as  xgb_final.pkl  (deployed in this Streamlit app)
    """, language="text")

    st.markdown("### Guidelines Followed ✅")
    items = [
        "Refund_Status dropped before splitting — prevents data leakage",
        "Split FIRST, clean after — per Sir Paolo's process flow",
        "LabelEncoders fit on training set only, applied to test separately",
        "SMOTE applied to training set only — never on test set",
        "Hyperparameters auto-tuned via RandomizedSearchCV — not fixed manually",
        "Primary metrics: AUC and F1-score — not accuracy",
        "Confusion matrix generated from test set predictions only",
        "Aklan now included — 17 records added from new cleaned dataset",
    ]
    for item in items:
        st.markdown(f"✅ {item}")

# ===========================================================================
# PREDICT
# ===========================================================================
elif page == "🔮 Predict Completion":
    st.title("🔮 Predict MSME Project Completion")
    if not model_loaded:
        st.error("Model not loaded. Please run SHI_XGBoost_Final.py first.")
        st.stop()

    st.markdown("Enter the MSME details below and click **Predict**.")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        province  = st.selectbox("🏙️ Province",
            ["Aklan","Antique","Capiz","Guimaras","Iloilo","Negros"])
        sector    = st.selectbox("🏭 Sector", [
            "Food Processing","Furniture","Metals & Engineering",
            "Gifts, Decors, Handicrafts","Horticulture & Agriculture",
            "Others (grouped)"])
        ownership = st.selectbox("🏢 Type of Ownership",
            ["Single","Corporation","Cooperative","Partnership"])
    with c2:
        size         = st.selectbox("📏 Size of Enterprise",["micro","small","medium"])
        project_cost = st.number_input("💰 Project Cost (₱)",
            min_value=10000, max_value=10000000, value=400000, step=10000,
            help="Total approved project cost in Philippine Peso")
        has_prior    = st.radio("📋 Has Prior Funding?", [False, True],
            format_func=lambda x: "✅ Yes (2nd or more project)" if x else "❌ No (1st project)")

    st.divider()
    predict_btn = st.button("🔍 Predict Completion Status", use_container_width=True, type="primary")

    if predict_btn:
        inp  = preprocess_input(province, sector, ownership, size, project_cost, has_prior)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]
        p_c  = prob[1]
        p_nc = prob[0]

        st.divider()
        st.markdown("### Prediction Result")
        r1, r2 = st.columns(2)

        with r1:
            if pred == 1:
                st.success("## ✅ LIKELY TO COMPLETE")
                st.metric("Completion Probability",    f"{p_c:.1%}")
                st.metric("Non-Completion Probability",f"{p_nc:.1%}")
            else:
                st.error("## ⚠️ AT RISK — NOT COMPLETED")
                st.metric("Non-Completion Probability",f"{p_nc:.1%}")
                st.metric("Completion Probability",    f"{p_c:.1%}")

            confidence = max(p_c, p_nc)
            if confidence >= 0.75:
                st.info(f"📊 Model confidence: **High** ({confidence:.1%})")
            elif confidence >= 0.60:
                st.info(f"📊 Model confidence: **Moderate** ({confidence:.1%})")
            else:
                st.warning(f"📊 Model confidence: **Low** ({confidence:.1%}) — interpret with caution")

        with r2:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            colors  = ['#EF5350','#42A5F5']
            bars    = ax.barh(['Not Completed','Completed'],[p_nc, p_c], color=colors, height=0.5)
            for bar, val in zip(bars, [p_nc, p_c]):
                ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                        f'{val:.1%}', va='center', fontsize=11, fontweight='bold')
            ax.set_xlim(0, 1.2)
            ax.axvline(0.5, color='gray', linestyle='--', lw=1, alpha=0.7)
            ax.set_xlabel('Probability')
            ax.set_title('Prediction Probabilities', fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()
        st.markdown("**Input Summary**")
        st.dataframe(pd.DataFrame([{
            "Province"     : province,
            "Sector"       : sector,
            "Ownership"    : ownership,
            "Size"         : size.capitalize(),
            "Project Cost" : f"₱{project_cost:,.0f}",
            "Prior Funding": "Yes" if has_prior else "No",
            "Prediction"   : "✅ Completed" if pred==1 else "⚠️ Not Completed",
            "Probability"  : f"{p_c:.1%}" if pred==1 else f"{p_nc:.1%}"
        }]), use_container_width=True)

# ===========================================================================
# MODEL PERFORMANCE
# ===========================================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    if not model_loaded:
        st.error("Model not loaded.")
        st.stop()

    st.markdown("### Test Set Metrics")
    st.caption("All metrics evaluated on held-out test set (20% of data, never seen during training)")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ROC-AUC (Test)",     f"{test_auc:.4f}", help="Area under ROC curve on test set")
    c2.metric("Weighted F1 (Test)", f"{test_f1:.4f}",  help="Weighted F1-score on test set")
    c3.metric("F1 (Not Completed)", f"{f1_nc:.4f}",    help="F1 for minority class")
    c4.metric("CV AUC (5-fold)",    f"{cv_auc:.4f}",   help="Cross-validation AUC on training set")

    st.divider()
    st.markdown("### Best Hyperparameters (Auto-selected by RandomizedSearchCV)")
    params_clean = {k: round(float(v), 4) for k, v in best_params.items()}
    params_df    = pd.DataFrame(list(params_clean.items()), columns=['Parameter','Best Value'])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Adviser's Guidelines Followed")
    st.info("""
**Per Sir Paolo's consultation (June 25):**

• **Primary metric:** AUC and F1-score — accuracy was NOT used as the main metric

• **Hyperparameter tuning:** RandomizedSearchCV (80 iterations, 5-fold StratifiedKFold)
  — hyperparameters were NOT fixed manually

• **Data leakage prevention:**
  - Refund_Status dropped before splitting
  - Encoders fit on training set only, applied to test separately
  - SMOTE applied to training set only

• **Evaluation:** Confusion matrix and all metrics on held-out test set only

• **Class imbalance:** Addressed via SMOTE on training set only
    """)

# ===========================================================================
# FEATURE IMPORTANCE
# ===========================================================================
elif page == "📊 Feature Importance":
    st.title("📊 Feature Importance")
    if not model_loaded:
        st.error("Model not loaded.")
        st.stop()

    st.markdown("Features ranked by their contribution to XGBoost predictions.")
    st.divider()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ['#1565C0' if i < 3 else '#42A5F5'
               for i in range(len(importance_df)-1, -1, -1)]
    bars    = ax.barh(importance_df['Feature'][::-1],
                      importance_df['Importance'][::-1], color=colors)
    for bar, val in zip(bars, importance_df['Importance'][::-1]):
        ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Importance Score')
    ax.set_title('XGBoost Feature Importance', fontsize=13)
    top3    = mpatches.Patch(color='#1565C0', label='Top 3 features')
    others  = mpatches.Patch(color='#42A5F5', label='Other features')
    ax.legend(handles=[top3, others], loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.dataframe(importance_df, use_container_width=True, hide_index=True)

    st.divider()
    top_feat = importance_df.iloc[0]['Feature']
    st.info(f"**Top feature: {top_feat}** — this variable contributes most to predicting whether an MSME completes their project.")

# ===========================================================================
# ABOUT
# ===========================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    ### MSME Project Completion Predictor
    **DOST SETUP 4.0 iFund Program — Western Visayas**

    ---

    **Model:** XGBoost (Extreme Gradient Boosting)

    **Dataset:** 321 MSME records across **6 provinces**
    (Aklan, Antique, Capiz, Guimaras, Iloilo, Negros Occidental)

    **Target Variable:** Completion Status (Completed / Not Completed)

    **Predictive Features:**
    | Feature | Description |
    |---|---|
    | Province | Where the MSME is located |
    | Sector | Industry sector (Food Processing, Furniture, etc.) |
    | Type of Ownership | Single, Corporation, Cooperative, Partnership |
    | Size of Enterprise | Micro, Small, Medium |
    | Project Cost | Total approved project cost (₱) |
    | Has Prior Funding | Whether the MSME had a previous DOST project |

    **Engineered Features:**
    | Feature | Description |
    |---|---|
    | Cost to Size Ratio | Project cost relative to enterprise size |
    | Log Project Cost | Log-transformed cost (reduces skewness) |

    ---

    **References:**
    - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD '16*.
    - Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.

    ---
    **Note:** Aklan is included in this version of the model (17 records).
    Model trained and evaluated following CRISP-DM framework.
    """)
