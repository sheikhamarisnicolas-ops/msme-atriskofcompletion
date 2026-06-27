"""
Streamlit App — XGBoost MSME Project Completion Predictor
DOST SETUP 4.0 iFund Program, Western Visayas

HOW TO RUN:
  streamlit run SHI_XGBoost_Streamlit_Final.py

NOTE: No pickle file needed. The model trains automatically on startup
      using MSME_data_cleaned.xlsx in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (f1_score, roc_auc_score, confusion_matrix,
                                      classification_report, ConfusionMatrixDisplay,
                                      roc_curve, precision_recall_curve)
from imblearn.over_sampling   import SMOTE
from xgboost                  import XGBClassifier
from scipy.stats              import randint, uniform

st.set_page_config(
    page_title = "MSME Completion Predictor",
    page_icon  = "📊",
    layout     = "wide"
)

RANDOM_STATE = 42

# ===========================================================================
# TRAIN MODEL (cached so it only runs once per session)
# ===========================================================================
@st.cache_resource
def train_model():
    # --- Load ---
    df_raw = pd.read_excel('MSME_data_cleaned.xlsx')

    # --- Drop leakage columns FIRST ---
    df = df_raw.drop(columns=['Beneficiary_Name', 'Year', 'Refund_Status']).copy()

    FEATURES = ['Province', 'Sector', 'type_of_ownership',
                'size_of_enterprise', 'Project_Cost', 'Has_Prior_Funding']
    X = df[FEATURES].copy()
    y = (df['Completion_Status'] == 'Completed').astype(int)

    # --- Normalize sector ---
    sector_map = {
        'Agriculture/Marine/Aquaculture' : 'Horticulture & Agriculture',
        'Horticulture and Agriculture'   : 'Horticulture & Agriculture',
    }
    X['Sector'] = X['Sector'].replace(sector_map)
    X['Sector'] = X['Sector'].where(
        ~X['Sector'].str.startswith('Others', na=False), 'Others (grouped)')

    # --- Split FIRST ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    # --- Encode (fit on TRAIN only) ---
    CAT_COLS = ['Province', 'Sector', 'type_of_ownership', 'size_of_enterprise']
    encoders    = {}
    X_train_enc = X_train.copy()
    X_test_enc  = X_test.copy()

    for col in CAT_COLS:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train[col].astype(str))
        X_test_enc[col]  = X_test[col].astype(str).map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1)
        encoders[col] = le

    X_train_enc['Has_Prior_Funding'] = X_train_enc['Has_Prior_Funding'].astype(int)
    X_test_enc['Has_Prior_Funding']  = X_test_enc['Has_Prior_Funding'].astype(int)

    # --- Feature engineering (separately) ---
    size_num_map = {'micro': 1, 'small': 2, 'medium': 3}
    def add_features(X_raw, X_enc):
        X_enc = X_enc.copy()
        sn = X_raw['size_of_enterprise'].str.lower().map(size_num_map)
        X_enc['Cost_to_Size_Ratio'] = X_raw['Project_Cost'].values / sn.values
        X_enc['Log_Project_Cost']   = np.log1p(X_raw['Project_Cost'].values)
        return X_enc

    X_train_enc = add_features(X_train, X_train_enc)
    X_test_enc  = add_features(X_test,  X_test_enc)

    # --- SMOTE on train only ---
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)

    # --- RandomizedSearchCV ---
    param_dist = {
        'n_estimators'    : randint(200, 600),
        'max_depth'       : randint(3, 8),
        'learning_rate'   : uniform(0.01, 0.2),
        'subsample'       : uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 6),
        'gamma'           : uniform(0, 0.3),
        'reg_alpha'       : uniform(0, 0.5),
        'reg_lambda'      : uniform(0.5, 2.0),
    }
    xgb = XGBClassifier(objective='binary:logistic', eval_metric='auc',
                        use_label_encoder=False, tree_method='hist',
                        random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(xgb, param_dist, n_iter=80, scoring='roc_auc',
                            cv=cv, n_jobs=-1, verbose=0, random_state=RANDOM_STATE)
    rs.fit(X_train_sm, y_train_sm)

    best_model   = rs.best_estimator_
    y_pred       = best_model.predict(X_test_enc)
    y_pred_proba = best_model.predict_proba(X_test_enc)[:, 1]

    test_auc = roc_auc_score(y_test, y_pred_proba)
    f1_w     = f1_score(y_test, y_pred, average='weighted')
    f1_nc    = f1_score(y_test, y_pred, pos_label=0)
    f1_c     = f1_score(y_test, y_pred, pos_label=1)

    importance_df = pd.DataFrame({
        'Feature'   : X_train_enc.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    return {
        'model'        : best_model,
        'encoders'     : encoders,
        'sector_map'   : sector_map,
        'size_num_map' : size_num_map,
        'feature_cols' : list(X_train_enc.columns),
        'best_params'  : rs.best_params_,
        'cv_auc'       : rs.best_score_,
        'test_auc'     : test_auc,
        'test_f1'      : f1_w,
        'f1_nc'        : f1_nc,
        'f1_c'         : f1_c,
        'importance_df': importance_df,
        'y_test'       : y_test,
        'y_pred'       : y_pred,
        'y_pred_proba' : y_pred_proba,
        'X_train_shape': X_train_enc.shape,
        'X_test_shape' : X_test_enc.shape,
    }

# ===========================================================================
# PREPROCESS INPUT (same steps as training)
# ===========================================================================
def preprocess_input(art, province, sector, ownership, size, cost, has_prior):
    sector = art['sector_map'].get(sector, sector)
    if sector.startswith('Others'):
        sector = 'Others (grouped)'

    row = pd.DataFrame([{
        'Province'          : province,
        'Sector'            : sector,
        'type_of_ownership' : ownership,
        'size_of_enterprise': size,
        'Project_Cost'      : cost,
        'Has_Prior_Funding' : int(has_prior)
    }])

    for col in ['Province','Sector','type_of_ownership','size_of_enterprise']:
        le  = art['encoders'][col]
        val = str(row[col].iloc[0])
        row[col] = le.transform([val])[0] if val in le.classes_ else -1

    size_num = art['size_num_map'].get(size.lower(), 1)
    row['Cost_to_Size_Ratio'] = cost / size_num
    row['Log_Project_Cost']   = np.log1p(cost)

    return row[art['feature_cols']]

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

# ===========================================================================
# LOAD MODEL WITH SPINNER
# ===========================================================================
with st.spinner("🔄 Training XGBoost model... please wait (runs once)"):
    try:
        art          = train_model()
        model_loaded = True
        with st.sidebar:
            st.divider()
            st.success("✅ Model ready")
            st.caption(f"CV AUC : {art['cv_auc']:.4f}\nTest AUC: {art['test_auc']:.4f}")
    except FileNotFoundError:
        model_loaded = False
        st.error("❌ `MSME_data_cleaned.xlsx` not found. Place it in the same folder as this app.")
        st.stop()

# ===========================================================================
# HOME
# ===========================================================================
if page == "🏠 Home":
    st.title("📊 MSME Project Completion Predictor")
    st.subheader("DOST SETUP 4.0 iFund Program — Western Visayas")
    st.markdown("Predicts whether an MSME beneficiary will **complete** their technology project using **XGBoost**.")
    st.divider()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total records", "321")
    c2.metric("Completed",     "223 (69.5%)")
    c3.metric("Not Completed", "98 (30.5%)")
    c4.metric("Provinces",     "6 incl. Aklan")

    st.divider()
    st.markdown("### Process Flow (per Adviser Sir Paolo)")
    st.code("""
Raw Data (321 records)
  ↓
Drop leakage columns — Refund_Status, Beneficiary_Name, Year
  ↓
Train-Test Split  80/20  (stratified on Completion_Status)
  ↓
Normalize & Encode → TRAIN and TEST separately (no leakage)
  ↓
Feature Engineering → Cost_to_Size_Ratio, Log_Project_Cost
  ↓
SMOTE → training set only (fixes class imbalance)
  ↓
RandomizedSearchCV → auto-tune hyperparameters (optimize AUC)
  ↓
Evaluate on TEST SET only → AUC, F1, Confusion Matrix
    """, language="text")

    st.markdown("### Guidelines Followed ✅")
    for item in [
        "Refund_Status dropped before splitting — prevents data leakage",
        "Split FIRST, clean after — per Sir Paolo's process flow",
        "LabelEncoders fit on training set only, applied to test separately",
        "SMOTE applied to training set only — never on test set",
        "Hyperparameters auto-tuned via RandomizedSearchCV — not fixed manually",
        "Primary metrics: AUC and F1-score — not accuracy",
        "Confusion matrix generated from test set predictions only",
        "Aklan now included — 17 records added from new cleaned dataset",
    ]:
        st.markdown(f"✅ {item}")

# ===========================================================================
# PREDICT
# ===========================================================================
elif page == "🔮 Predict Completion":
    st.title("🔮 Predict MSME Project Completion")
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
        size         = st.selectbox("📏 Size of Enterprise", ["micro","small","medium"])
        project_cost = st.number_input("💰 Project Cost (₱)",
            min_value=10000, max_value=10000000, value=400000, step=10000)
        has_prior    = st.radio("📋 Has Prior Funding?", [False, True],
            format_func=lambda x: "✅ Yes (2nd or more project)" if x else "❌ No (1st project)")

    st.divider()
    if st.button("🔍 Predict Completion Status", use_container_width=True, type="primary"):
        inp  = preprocess_input(art, province, sector, ownership, size, project_cost, has_prior)
        pred = art['model'].predict(inp)[0]
        prob = art['model'].predict_proba(inp)[0]
        p_c, p_nc = prob[1], prob[0]

        st.divider()
        st.markdown("### Prediction Result")
        r1, r2 = st.columns(2)

        with r1:
            if pred == 1:
                st.success("## ✅ LIKELY TO COMPLETE")
                st.metric("Completion Probability",     f"{p_c:.1%}")
                st.metric("Non-Completion Probability", f"{p_nc:.1%}")
            else:
                st.error("## ⚠️ AT RISK — NOT COMPLETED")
                st.metric("Non-Completion Probability", f"{p_nc:.1%}")
                st.metric("Completion Probability",     f"{p_c:.1%}")

            confidence = max(p_c, p_nc)
            if confidence >= 0.75:
                st.info(f"📊 Model confidence: **High** ({confidence:.1%})")
            elif confidence >= 0.60:
                st.info(f"📊 Model confidence: **Moderate** ({confidence:.1%})")
            else:
                st.warning(f"📊 Model confidence: **Low** ({confidence:.1%}) — interpret with caution")

        with r2:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            bars = ax.barh(['Not Completed','Completed'], [p_nc, p_c],
                           color=['#EF5350','#42A5F5'], height=0.5)
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
        }]), use_container_width=True, hide_index=True)

# ===========================================================================
# MODEL PERFORMANCE
# ===========================================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.caption("All metrics evaluated on held-out test set — never seen during training.")

    st.markdown("### Test Set Metrics")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ROC-AUC (Test)",     f"{art['test_auc']:.4f}")
    c2.metric("Weighted F1 (Test)", f"{art['test_f1']:.4f}")
    c3.metric("F1 (Not Completed)", f"{art['f1_nc']:.4f}")
    c4.metric("CV AUC (5-fold)",    f"{art['cv_auc']:.4f}")

    st.divider()
    st.markdown("### Confusion Matrix (Test Set)")
    cm = confusion_matrix(art['y_test'], art['y_pred'])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Not Completed','Completed']).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('XGBoost — Confusion Matrix (Test Set)', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.markdown("### ROC Curve (Test Set)")
    fpr, tpr, _ = roc_curve(art['y_test'], art['y_pred_proba'])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f"XGBoost (AUC = {art['test_auc']:.3f})")
    ax.plot([0,1],[0,1], color='gray', linestyle='--', lw=1, label='Random Chance')
    ax.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — XGBoost (Test Set)')
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.markdown("### Best Hyperparameters (Auto-selected by RandomizedSearchCV)")
    params_df = pd.DataFrame(
        [(k, round(float(v), 4)) for k, v in art['best_params'].items()],
        columns=['Parameter','Best Value'])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.divider()
    st.info("""
**Per Sir Paolo's consultation (June 25):**
- Primary metric: AUC and F1-score — accuracy was NOT used
- Hyperparameters NOT fixed manually — auto-tuned via RandomizedSearchCV
- Refund_Status dropped before splitting — no data leakage
- Encoders fit on training set only, applied to test separately
- SMOTE on training set only — never on test set
- Confusion matrix on held-out test set only
    """)

# ===========================================================================
# FEATURE IMPORTANCE
# ===========================================================================
elif page == "📊 Feature Importance":
    st.title("📊 Feature Importance")
    st.markdown("Features ranked by their contribution to XGBoost predictions.")
    st.divider()

    imp = art['importance_df']
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#1565C0' if i < 3 else '#42A5F5'
              for i in range(len(imp)-1, -1, -1)]
    bars = ax.barh(imp['Feature'][::-1], imp['Importance'][::-1], color=colors)
    for bar, val in zip(bars, imp['Importance'][::-1]):
        ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Importance Score')
    ax.set_title('XGBoost Feature Importance', fontsize=13)
    top3   = mpatches.Patch(color='#1565C0', label='Top 3 features')
    others = mpatches.Patch(color='#42A5F5', label='Other features')
    ax.legend(handles=[top3, others], loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.dataframe(imp, use_container_width=True, hide_index=True)
    st.info(f"**Top feature: {imp.iloc[0]['Feature']}** — contributes most to predicting MSME project completion.")

# ===========================================================================
# ABOUT
# ===========================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.markdown("""
    ### MSME Project Completion Predictor
    **DOST SETUP 4.0 iFund Program — Western Visayas**

    **Model:** XGBoost (Extreme Gradient Boosting)

    **Dataset:** 321 MSME records | 6 provinces (Aklan, Antique, Capiz, Guimaras, Iloilo, Negros Occidental)

    **Predictive Features:**
    | Feature | Description |
    |---|---|
    | Province | Where the MSME is located |
    | Sector | Industry sector |
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
    - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
    - Chawla, N. V. et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR, 16*, 321–357.

    ---
    *Model trained following CRISP-DM framework. Aklan included in this version.*
    """)
