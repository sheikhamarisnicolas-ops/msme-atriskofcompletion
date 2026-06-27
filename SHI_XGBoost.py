"""
Streamlit App — MSME At Risk of Completion Predictor
DOST SETUP 4.0 iFund Program, Western Visayas
XGBoost | Threshold Tuning | Risk Tiers | Feature Explanation

HOW TO RUN:
  streamlit run SHI_XGBoost_Streamlit_Final.py
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
                                      ConfusionMatrixDisplay, roc_curve,
                                      precision_recall_curve)
from imblearn.over_sampling   import SMOTE
from xgboost                  import XGBClassifier
from scipy.stats              import randint, uniform

# ===========================================================================
# PALETTE
# ===========================================================================
C1  = "#2C5EAD"
C2  = "#1591DC"
C3  = "#4BB8FA"
C4  = "#C4E2F5"
RED = "#E05A5A"
ORG = "#E09A2A"
GRN = "#2EAD72"

st.set_page_config(page_title="MSME Risk Predictor", page_icon="📊", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background-color: #F7FBFF; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {C1} 0%, {C2} 100%);
    }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    [data-testid="metric-container"] {{
        background: white;
        border: 1px solid {C4};
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(44,94,173,0.08);
    }}
    [data-testid="metric-container"] label {{ color: {C1} !important; font-size: 12px; }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {C1} !important; font-size: 22px; font-weight: 600;
    }}
    h2, h3 {{ color: {C1} !important; }}
    .stButton > button {{
        background: linear-gradient(90deg, {C1}, {C2});
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.6rem 1.2rem;
    }}
    .stButton > button:hover {{ background: linear-gradient(90deg, {C2}, {C3}); color: white; }}
    .card {{
        background: white; border-radius: 10px; padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(44,94,173,0.08); margin-bottom: 1rem;
        border: 1px solid {C4};
    }}
    .page-title {{ color: {C1}; font-size: 26px; font-weight: 700; margin-bottom: 4px; }}
    .page-sub   {{ color: #6B8BBE; font-size: 14px; margin-bottom: 1.5rem; }}
    .badge-high   {{ background:#FDECEA; color:{RED}; border:1px solid {RED};
                     border-radius:20px; padding:4px 14px; font-weight:700; font-size:13px; }}
    .badge-medium {{ background:#FEF3E2; color:{ORG}; border:1px solid {ORG};
                     border-radius:20px; padding:4px 14px; font-weight:700; font-size:13px; }}
    .badge-low    {{ background:#E6F9F1; color:{GRN}; border:1px solid {GRN};
                     border-radius:20px; padding:4px 14px; font-weight:700; font-size:13px; }}
    .factor-bar {{ height:8px; border-radius:4px; background:{C3}; margin:4px 0 10px 0; }}
</style>
""", unsafe_allow_html=True)

RANDOM_STATE = 42

# ===========================================================================
# TRAIN MODEL
# ===========================================================================
@st.cache_resource
def train_model():
    df_raw = pd.read_excel('MSME_data_cleaned.xlsx')
    df = df_raw.drop(columns=['Beneficiary_Name','Year','Refund_Status']).copy()

    FEATURES = ['Province','Sector','type_of_ownership',
                'size_of_enterprise','Project_Cost','Has_Prior_Funding']
    X = df[FEATURES].copy()
    y = (df['Completion_Status'] == 'Completed').astype(int)

    sector_map = {
        'Agriculture/Marine/Aquaculture':'Horticulture & Agriculture',
        'Horticulture and Agriculture'  :'Horticulture & Agriculture',
    }
    X['Sector'] = X['Sector'].replace(sector_map)
    X['Sector'] = X['Sector'].where(
        ~X['Sector'].str.startswith('Others', na=False), 'Others (grouped)')

    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    CAT_COLS = ['Province','Sector','type_of_ownership','size_of_enterprise']
    encoders = {}
    Xtr = X_train.copy(); Xte = X_test.copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        Xtr[col] = le.fit_transform(X_train[col].astype(str))
        Xte[col] = X_test[col].astype(str).map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1)
        encoders[col] = le

    Xtr['Has_Prior_Funding'] = Xtr['Has_Prior_Funding'].astype(int)
    Xte['Has_Prior_Funding'] = Xte['Has_Prior_Funding'].astype(int)

    size_num_map = {'micro':1,'small':2,'medium':3}
    def add_feat(Xr, Xe):
        Xe = Xe.copy()
        sn = Xr['size_of_enterprise'].str.lower().map(size_num_map)
        Xe['Cost_to_Size_Ratio'] = Xr['Project_Cost'].values / sn.values
        Xe['Log_Project_Cost']   = np.log1p(Xr['Project_Cost'].values)
        return Xe

    Xtr = add_feat(X_train, Xtr); Xte = add_feat(X_test, Xte)
    Xtr_sm, ytr_sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5).fit_resample(Xtr, y_train)

    param_dist = {
        'n_estimators'    : randint(200,600),
        'max_depth'       : randint(3,8),
        'learning_rate'   : uniform(0.01,0.2),
        'subsample'       : uniform(0.6,0.4),
        'colsample_bytree': uniform(0.6,0.4),
        'min_child_weight': randint(1,6),
        'gamma'           : uniform(0,0.3),
        'reg_alpha'       : uniform(0,0.5),
        'reg_lambda'      : uniform(0.5,2.0),
    }
    xgb = XGBClassifier(objective='binary:logistic', eval_metric='auc',
                        use_label_encoder=False, tree_method='hist',
                        random_state=RANDOM_STATE)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rs  = RandomizedSearchCV(xgb, param_dist, n_iter=80, scoring='roc_auc',
                             cv=cv, n_jobs=-1, verbose=0, random_state=RANDOM_STATE)
    rs.fit(Xtr_sm, ytr_sm)

    bm       = rs.best_estimator_
    y_pred_p = bm.predict_proba(Xte)[:,1]

    # --- THRESHOLD TUNING ---
    # Find threshold that achieves at least 75% recall on Not Completed (minority)
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_pred_p, pos_label=0)
    best_thresh = 0.5
    for p, r, t in zip(prec_arr, rec_arr, thresh_arr):
        if r >= 0.75:
            best_thresh = float(1 - t)
            break

    y_pred_tuned = (y_pred_p >= (1 - best_thresh)).astype(int)

    test_auc = roc_auc_score(y_test, y_pred_p)
    f1_w     = f1_score(y_test, y_pred_tuned, average='weighted')
    f1_nc    = f1_score(y_test, y_pred_tuned, pos_label=0)
    f1_c     = f1_score(y_test, y_pred_tuned, pos_label=1)

    imp = pd.DataFrame({'Feature':Xtr.columns,'Importance':bm.feature_importances_})\
            .sort_values('Importance',ascending=False).reset_index(drop=True)

    return dict(
        model=bm, encoders=encoders, sector_map=sector_map,
        size_num_map=size_num_map, feature_cols=list(Xtr.columns),
        best_params=rs.best_params_, cv_auc=rs.best_score_,
        test_auc=test_auc, test_f1=f1_w, f1_nc=f1_nc, f1_c=f1_c,
        importance_df=imp, y_test=y_test,
        y_pred=y_pred_tuned, y_pred_proba=y_pred_p,
        threshold=best_thresh,
    )

# ===========================================================================
# PREPROCESS + PREDICT
# ===========================================================================
def preprocess_input(art, province, sector, ownership, size, cost, has_prior):
    sector = art['sector_map'].get(sector, sector)
    if sector.startswith('Others'): sector = 'Others (grouped)'
    row = pd.DataFrame([{
        'Province':province,'Sector':sector,
        'type_of_ownership':ownership,'size_of_enterprise':size,
        'Project_Cost':cost,'Has_Prior_Funding':int(has_prior)}])
    for col in ['Province','Sector','type_of_ownership','size_of_enterprise']:
        le  = art['encoders'][col]; val = str(row[col].iloc[0])
        row[col] = le.transform([val])[0] if val in le.classes_ else -1
    sn = art['size_num_map'].get(size.lower(),1)
    row['Cost_to_Size_Ratio'] = cost / sn
    row['Log_Project_Cost']   = np.log1p(cost)
    return row[art['feature_cols']]

def get_risk_tier(p_completed):
    pct = p_completed * 100
    if pct < 44:   return "High",   RED, "🔴"
    elif pct < 65: return "Medium", ORG, "🟡"
    else:          return "Low",    GRN, "🟢"

def get_top_factors(art, n=5):
    imp = art['importance_df']
    feat_labels = {
        'Province'          : 'Province',
        'Sector'            : 'Sector',
        'type_of_ownership' : 'Type of Ownership',
        'size_of_enterprise': 'Size of Enterprise',
        'Project_Cost'      : 'Project Cost',
        'Has_Prior_Funding' : 'Has Prior Funding',
        'Cost_to_Size_Ratio': 'Cost-to-Size Ratio',
        'Log_Project_Cost'  : 'Log Project Cost',
    }
    rows = []
    for _, r in imp.head(n).iterrows():
        rows.append((feat_labels.get(r['Feature'], r['Feature']), r['Importance']))
    return rows

def make_fig(w=4.5, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_edgecolor(C4)
    ax.tick_params(colors='#555')
    ax.xaxis.label.set_color('#555'); ax.yaxis.label.set_color('#555')
    ax.title.set_color(C1)
    return fig, ax

# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("<div style='font-size:22px;font-weight:700;margin-bottom:2px'>📊 MSME Risk</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;opacity:0.85;margin-bottom:1rem'>DOST SETUP 4.0 iFund<br>Western Visayas</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "🏠  Home",
        "⚠️  Risk Assessment",
        "📈  Model Performance",
        "📊  Feature Importance",
        "ℹ️  About"
    ], label_visibility="collapsed")

# ===========================================================================
# TRAIN
# ===========================================================================
with st.spinner("Setting up model..."):
    try:
        art = train_model()
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"<div style='font-size:12px;opacity:0.85'>Model ready ✅<br>CV AUC: {art['cv_auc']:.4f}<br>Test AUC: {art['test_auc']:.4f}<br>Threshold: {art['threshold']:.3f}</div>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("MSME_data_cleaned.xlsx not found. Upload it to your GitHub repo.")
        st.stop()

# ===========================================================================
# HOME
# ===========================================================================
if page == "🏠  Home":
    st.markdown("<div class='page-title'>MSME Project Completion Risk Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>DOST SETUP 4.0 iFund Program — Western Visayas &nbsp;|&nbsp; Model: XGBoost</div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records",  "321")
    c2.metric("Completed",      "223")
    c3.metric("Not Completed",  "98")
    c4.metric("Provinces",      "6")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>🎯 Objective</h3>
        <p style='color:#444;font-size:14px;line-height:1.7'>
        This tool uses <b>XGBoost</b> to predict whether an MSME beneficiary
        is <b>at risk of not completing</b> their technology project under the
        DOST SETUP 4.0 iFund Program. It is trained on historical data from
        321 enterprises across 6 provinces in Western Visayas.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>🏷️ Risk Tiers</h3>
        <table style='font-size:13px;width:100%;border-collapse:collapse'>
        <tr>
          <td style='padding:8px 0'><span class='badge-high'>🔴 High Risk</span></td>
          <td style='padding:8px;color:#444'>Completion probability below 44% — needs immediate attention</td>
        </tr>
        <tr>
          <td style='padding:8px 0'><span class='badge-medium'>🟡 Medium Risk</span></td>
          <td style='padding:8px;color:#444'>Completion probability 44–65% — monitor closely</td>
        </tr>
        <tr>
          <td style='padding:8px 0'><span class='badge-low'>🟢 Low Risk</span></td>
          <td style='padding:8px;color:#444'>Completion probability above 65% — on track</td>
        </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>⚙️ Model Pipeline</h3>
        <p style='color:#444;font-size:13px;line-height:1.9'>
        📌 Drop leakage columns (Refund_Status, Year, Beneficiary_Name)<br>
        📌 Train-Test Split 80/20 — stratified<br>
        📌 Encode Train and Test separately — no leakage<br>
        📌 Feature Engineering (Cost-to-Size Ratio, Log Cost)<br>
        📌 SMOTE on training set only<br>
        📌 RandomizedSearchCV — 80 iterations, optimize AUC<br>
        📌 Threshold tuning — optimize recall for at-risk cases<br>
        📌 Evaluate on test set (AUC, F1, Confusion Matrix)
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>📊 Model Results</h3>
        <table style='font-size:13px;width:100%;border-collapse:collapse;color:#444'>
        <tr style='border-bottom:1px solid {C4}'><td style='padding:6px 0'><b>Cross-Validation AUC</b></td><td style='padding:6px;color:{C1};font-weight:600'>{art['cv_auc']:.4f}</td></tr>
        <tr style='border-bottom:1px solid {C4}'><td style='padding:6px 0'><b>Test AUC</b></td><td style='padding:6px;color:{C1};font-weight:600'>{art['test_auc']:.4f}</td></tr>
        <tr style='border-bottom:1px solid {C4}'><td style='padding:6px 0'><b>Weighted F1</b></td><td style='padding:6px;color:{C1};font-weight:600'>{art['test_f1']:.4f}</td></tr>
        <tr style='border-bottom:1px solid {C4}'><td style='padding:6px 0'><b>F1 (Not Completed)</b></td><td style='padding:6px;color:{C1};font-weight:600'>{art['f1_nc']:.4f}</td></tr>
        <tr><td style='padding:6px 0'><b>Decision Threshold</b></td><td style='padding:6px;color:{C1};font-weight:600'>{art['threshold']:.3f}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

# ===========================================================================
# RISK ASSESSMENT
# ===========================================================================
elif page == "⚠️  Risk Assessment":
    st.markdown("<div class='page-title'>⚠️ At Risk of Completion Assessment</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Enter MSME details to assess project completion risk based on historical data.</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        province  = st.selectbox("Province", ["Aklan","Antique","Capiz","Guimaras","Iloilo","Negros"])
        sector    = st.selectbox("Sector", ["Food Processing","Furniture","Metals & Engineering",
                                             "Gifts, Decors, Handicrafts","Horticulture & Agriculture","Others (grouped)"])
    with c2:
        ownership = st.selectbox("Type of Ownership", ["Single","Corporation","Cooperative","Partnership"])
        size      = st.selectbox("Size of Enterprise", ["micro","small","medium"])
    with c3:
        cost      = st.number_input("Project Cost (₱)", min_value=10000, max_value=10000000,
                                    value=400000, step=10000)
        has_prior = st.radio("Has Prior Funding?", [False, True],
                             format_func=lambda x: "Yes — 2nd or more project" if x else "No — 1st project")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Assess Completion Risk", use_container_width=True):
        inp      = preprocess_input(art, province, sector, ownership, size, cost, has_prior)
        prob     = art['model'].predict_proba(inp)[0]
        p_c, p_nc = prob[1], prob[0]
        pred     = int(p_nc >= (1 - art['threshold']))  # 1=at risk using tuned threshold
        tier, tier_color, tier_icon = get_risk_tier(p_c)
        factors  = get_top_factors(art, n=5)

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1.2, 1, 1.2])

        # --- Result Card ---
        with r1:
            if pred == 0:
                bg, border, label, desc = C4, C1, "✅ LOW RISK", "Likely to complete their project."
                prob_val, prob_label = p_c, "Completion Probability"
            else:
                bg, border, label, desc = "#FDECEA", RED, "⚠️ AT RISK", "Likely to NOT complete their project."
                prob_val, prob_label = p_nc, "Non-Completion Probability"

            st.markdown(f"""
            <div style='background:{bg};border-left:5px solid {border};border-radius:8px;padding:1.2rem 1.5rem;margin-bottom:1rem'>
                <div style='font-size:18px;font-weight:700;color:{border}'>{label}</div>
                <div style='font-size:13px;color:#444;margin-top:4px'>{desc}</div>
                <div style='margin-top:14px;font-size:30px;font-weight:700;color:{border}'>{prob_val:.1%}</div>
                <div style='font-size:12px;color:#888'>{prob_label}</div>
                <div style='margin-top:12px'>
                    <span class='badge-{"high" if tier=="High" else "medium" if tier=="Medium" else "low"}'>
                    {tier_icon} {tier} Risk
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence
            conf = max(p_c, p_nc)
            conf_label = "High" if conf >= 0.75 else "Moderate" if conf >= 0.60 else "Low"
            conf_color = GRN if conf >= 0.75 else C2 if conf >= 0.60 else ORG
            st.markdown(f"<div style='font-size:12px;color:{conf_color}'>📊 Model confidence: <b>{conf_label}</b> ({conf:.1%})</div>", unsafe_allow_html=True)

        # --- Probability Bar ---
        with r2:
            fig, ax = make_fig(3.5, 3)
            ax.barh(['Not Completed','Completed'], [p_nc, p_c],
                    color=[RED, C2], height=0.45)
            for val, y in zip([p_nc, p_c], [0, 1]):
                ax.text(val+0.01, y, f'{val:.1%}', va='center', fontsize=10, fontweight='600', color='#333')
            ax.set_xlim(0, 1.2)
            ax.axvline(1 - art['threshold'], color='gray', linestyle='--', lw=1, alpha=0.7)
            ax.set_xlabel('Probability', fontsize=10)
            ax.set_title('Risk Probability', fontsize=11, fontweight='600')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close()
            st.caption(f"Dashed line = decision threshold ({1-art['threshold']:.3f})")

        # --- Top Contributing Factors ---
        with r3:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C1};margin-bottom:10px'>🔍 Top Contributing Factors</div>", unsafe_allow_html=True)
            max_imp = factors[0][1] if factors else 1
            for name, imp_val in factors:
                bar_w = int((imp_val / max_imp) * 100)
                st.markdown(f"""
                <div style='font-size:12px;color:#444;margin-bottom:2px'><b>{name}</b> <span style='color:#888;float:right'>{imp_val:.3f}</span></div>
                <div style='background:{C4};border-radius:4px;height:8px;margin-bottom:10px'>
                    <div style='background:{C2};width:{bar_w}%;height:8px;border-radius:4px'></div>
                </div>
                """, unsafe_allow_html=True)
            st.caption("Global feature importance from XGBoost model.")

        # --- Input Summary ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:13px;font-weight:600;color:{C1};margin-bottom:6px'>Input Summary</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Province":province,"Sector":sector,"Ownership":ownership,
            "Size":size.capitalize(),"Project Cost":f"₱{cost:,.0f}",
            "Prior Funding":"Yes" if has_prior else "No",
            "Risk Tier":f"{tier_icon} {tier}",
            "Probability":f"{prob_val:.1%}"
        }]), use_container_width=True, hide_index=True)

# ===========================================================================
# MODEL PERFORMANCE
# ===========================================================================
elif page == "📈  Model Performance":
    st.markdown("<div class='page-title'>📈 Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Evaluated on held-out test set — 20% of data, never seen during training.</div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ROC-AUC (Test)",     f"{art['test_auc']:.4f}")
    c2.metric("Weighted F1",        f"{art['test_f1']:.4f}")
    c3.metric("F1 (Not Completed)", f"{art['f1_nc']:.4f}")
    c4.metric("CV AUC (5-fold)",    f"{art['cv_auc']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<div style='font-size:15px;font-weight:600;color:{C1};margin-bottom:8px'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = confusion_matrix(art['y_test'], art['y_pred'])
        fig, ax = make_fig(4, 3.2)
        ConfusionMatrixDisplay(cm, display_labels=['Not Completed','Completed']).plot(
            ax=ax, cmap='Blues', colorbar=False)
        ax.set_title('Confusion Matrix — Test Set', fontsize=11, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close()

    with col2:
        st.markdown(f"<div style='font-size:15px;font-weight:600;color:{C1};margin-bottom:8px'>ROC Curve</div>", unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(art['y_test'], art['y_pred_proba'])
        fig, ax = make_fig(4, 3.2)
        ax.plot(fpr, tpr, color=C2, lw=2, label=f"AUC = {art['test_auc']:.3f}")
        ax.plot([0,1],[0,1], color='#CCC', linestyle='--', lw=1, label='Random')
        ax.fill_between(fpr, tpr, alpha=0.08, color=C3)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title('ROC Curve — Test Set', fontsize=11, fontweight='600')
        ax.legend(fontsize=9, framealpha=0.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='card'>
    <h3 style='margin-top:0'>🎯 Threshold Tuning</h3>
    <p style='font-size:13px;color:#444;line-height:1.7'>
    The default decision threshold in classification is <b>0.5</b>. However, in this study,
    missing a truly at-risk MSME is more costly than a false alarm. The threshold was
    tuned to achieve at least <b>75% recall</b> on the Not Completed class, prioritizing
    the detection of at-risk beneficiaries for early DOST intervention.
    </p>
    <p style='font-size:14px;font-weight:600;color:{C1}'>Optimized threshold: {art['threshold']:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:15px;font-weight:600;color:{C1};margin-bottom:8px'>Best Hyperparameters — Auto-selected by RandomizedSearchCV</div>", unsafe_allow_html=True)
    params_df = pd.DataFrame(
        [(k, round(float(v),4)) for k,v in art['best_params'].items()],
        columns=['Parameter','Best Value'])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

# ===========================================================================
# FEATURE IMPORTANCE
# ===========================================================================
elif page == "📊  Feature Importance":
    st.markdown("<div class='page-title'>📊 Feature Importance</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Variables ranked by their contribution to XGBoost predictions.</div>", unsafe_allow_html=True)

    imp = art['importance_df']
    col1, col2 = st.columns([3, 2])

    with col1:
        fig, ax = make_fig(5.5, 4)
        colors = [C1 if i < 3 else C3 for i in range(len(imp)-1,-1,-1)]
        bars   = ax.barh(imp['Feature'][::-1], imp['Importance'][::-1], color=colors, height=0.55)
        for bar, val in zip(bars, imp['Importance'][::-1]):
            ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9, color='#333')
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.set_title('Feature Importance', fontsize=12, fontweight='600')
        top3   = mpatches.Patch(color=C1, label='Top 3')
        others = mpatches.Patch(color=C3, label='Others')
        ax.legend(handles=[top3,others], fontsize=9, framealpha=0.5)
        ax.set_xlim(0, imp['Importance'].max() * 1.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close()

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(imp, use_container_width=True, hide_index=True)
        top = imp.iloc[0]['Feature']
        st.markdown(f"""
        <div class='card' style='margin-top:1rem'>
        <div style='font-size:13px;color:{C1};font-weight:600'>Top Feature</div>
        <div style='font-size:18px;font-weight:700;color:{C2};margin:4px 0'>{top}</div>
        <div style='font-size:12px;color:#666'>This variable contributes most
        to predicting MSME project completion risk.</div>
        </div>
        """, unsafe_allow_html=True)

# ===========================================================================
# ABOUT
# ===========================================================================
elif page == "ℹ️  About":
    st.markdown("<div class='page-title'>ℹ️ About</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>MSME Project Completion Risk Predictor — DOST SETUP 4.0 iFund Program</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>Dataset</h3>
        <p style='font-size:13px;color:#444;line-height:1.8'>
        <b>Records:</b> 321 MSME enterprises<br>
        <b>Provinces:</b> Aklan, Antique, Capiz, Guimaras, Iloilo, Negros Occidental<br>
        <b>Source:</b> DOST SETUP 4.0 iFund Program<br>
        <b>Target:</b> Completion Status (Completed / Not Completed)
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>Predictive Features</h3>
        <table style='font-size:13px;color:#444;width:100%;border-collapse:collapse'>
        <tr><td style='padding:5px 0;border-bottom:1px solid {C4}'><b>Province</b></td><td style='padding:5px 8px;border-bottom:1px solid {C4}'>Location of MSME</td></tr>
        <tr><td style='padding:5px 0;border-bottom:1px solid {C4}'><b>Sector</b></td><td style='padding:5px 8px;border-bottom:1px solid {C4}'>Industry sector</td></tr>
        <tr><td style='padding:5px 0;border-bottom:1px solid {C4}'><b>Type of Ownership</b></td><td style='padding:5px 8px;border-bottom:1px solid {C4}'>Single, Corp, Coop, Partnership</td></tr>
        <tr><td style='padding:5px 0;border-bottom:1px solid {C4}'><b>Size of Enterprise</b></td><td style='padding:5px 8px;border-bottom:1px solid {C4}'>Micro, Small, Medium</td></tr>
        <tr><td style='padding:5px 0;border-bottom:1px solid {C4}'><b>Project Cost</b></td><td style='padding:5px 8px;border-bottom:1px solid {C4}'>Total approved cost (₱)</td></tr>
        <tr><td style='padding:5px 0'><b>Has Prior Funding</b></td><td style='padding:5px 8px'>Previous DOST project</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>Model</h3>
        <p style='font-size:13px;color:#444;line-height:1.8'>
        <b>Algorithm:</b> XGBoost (Extreme Gradient Boosting)<br>
        <b>Tuning:</b> RandomizedSearchCV, 80 iterations<br>
        <b>Validation:</b> 5-fold Stratified K-Fold<br>
        <b>Imbalance fix:</b> SMOTE on training set only<br>
        <b>Threshold:</b> Tuned for ≥75% recall on at-risk class<br>
        <b>Primary metric:</b> ROC-AUC and F1-Score
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>References</h3>
        <p style='font-size:13px;color:#444;line-height:1.8'>
        Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
        <i>Proceedings of the 22nd ACM SIGKDD.</i><br><br>
        Chawla, N. V. et al. (2002). SMOTE: Synthetic minority over-sampling technique.
        <i>Journal of Artificial Intelligence Research, 16</i>, 321–357.
        </p>
        </div>
        """, unsafe_allow_html=True)
