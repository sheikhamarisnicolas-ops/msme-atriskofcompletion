"""
Streamlit App — MSME At Risk of Completion Predictor
DOST SETUP 4.0 iFund Program, Western Visayas

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
                                      ConfusionMatrixDisplay, roc_curve)
from imblearn.over_sampling   import SMOTE
from xgboost                  import XGBClassifier
from scipy.stats              import randint, uniform

# ===========================================================================
# PALETTE & THEME
# ===========================================================================
C1  = "#2C5EAD"   # dark blue
C2  = "#1591DC"   # medium blue
C3  = "#4BB8FA"   # light blue
C4  = "#C4E2F5"   # very light blue
RED = "#E05A5A"   # at-risk red

st.set_page_config(
    page_title = "MSME Risk Predictor",
    page_icon  = "📊",
    layout     = "wide"
)

# Custom CSS
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{ background-color: #F7FBFF; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {C1} 0%, {C2} 100%);
    }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    [data-testid="stSidebar"] .stRadio label {{ color: white !important; }}

    /* Metric cards */
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

    /* Section headers */
    h2, h3 {{ color: {C1} !important; }}

    /* Divider color */
    hr {{ border-color: {C4}; }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(90deg, {C1}, {C2});
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
    }}
    .stButton > button:hover {{
        background: linear-gradient(90deg, {C2}, {C3});
        color: white;
    }}

    /* Info / success / error boxes */
    .result-complete {{
        background: {C4};
        border-left: 5px solid {C1};
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .result-atrisk {{
        background: #FDE8E8;
        border-left: 5px solid {RED};
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .card {{
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(44,94,173,0.08);
        margin-bottom: 1rem;
        border: 1px solid {C4};
    }}
    .page-title {{
        color: {C1};
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }}
    .page-sub {{
        color: #6B8BBE;
        font-size: 14px;
        margin-bottom: 1.5rem;
    }}
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

    bm        = rs.best_estimator_
    y_pred    = bm.predict(Xte)
    y_pred_p  = bm.predict_proba(Xte)[:,1]
    test_auc  = roc_auc_score(y_test, y_pred_p)
    f1_w      = f1_score(y_test, y_pred, average='weighted')
    f1_nc     = f1_score(y_test, y_pred, pos_label=0)
    f1_c      = f1_score(y_test, y_pred, pos_label=1)
    imp       = pd.DataFrame({'Feature':Xtr.columns,'Importance':bm.feature_importances_})\
                  .sort_values('Importance',ascending=False).reset_index(drop=True)

    return dict(model=bm, encoders=encoders, sector_map=sector_map,
                size_num_map=size_num_map, feature_cols=list(Xtr.columns),
                best_params=rs.best_params_, cv_auc=rs.best_score_,
                test_auc=test_auc, test_f1=f1_w, f1_nc=f1_nc, f1_c=f1_c,
                importance_df=imp, y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_p)

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

def make_fig():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor(C4)
    ax.tick_params(colors='#444')
    ax.xaxis.label.set_color('#444')
    ax.yaxis.label.set_color('#444')
    ax.title.set_color(C1)
    return fig, ax

# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown(f"<div style='font-size:22px;font-weight:700;margin-bottom:4px'>📊 MSME Risk</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;opacity:0.85;margin-bottom:1.5rem'>DOST SETUP 4.0 iFund<br>Western Visayas</div>", unsafe_allow_html=True)
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
            st.markdown(f"<div style='font-size:12px;opacity:0.85'>Model ready ✅<br>CV AUC: {art['cv_auc']:.4f}<br>Test AUC: {art['test_auc']:.4f}</div>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("MSME_data_cleaned.xlsx not found. Upload it to your GitHub repo.")
        st.stop()

# ===========================================================================
# HOME
# ===========================================================================
if page == "🏠  Home":
    st.markdown(f"<div class='page-title'>MSME Project Completion Risk Predictor</div>", unsafe_allow_html=True)
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
        This tool uses <b>XGBoost</b> — a machine learning model — to predict whether
        an MSME beneficiary is <b>at risk of not completing</b> their technology project
        under the DOST SETUP 4.0 iFund Program. It is trained on historical data from
        321 enterprises across 6 provinces in Western Visayas.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>⚙️ Model Pipeline</h3>
        <p style='color:#444;font-size:13px;line-height:1.9'>
        📌 Drop leakage columns<br>
        📌 Train-Test Split 80/20 (stratified)<br>
        📌 Encode Train and Test separately<br>
        📌 SMOTE on training set only<br>
        📌 RandomizedSearchCV — optimize AUC<br>
        📌 Evaluate on test set (AUC, F1)
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='card'>
    <h3 style='margin-top:0'>✅ Key Design Decisions</h3>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;color:#444'>
        <div>✅ Refund_Status dropped — prevents data leakage</div>
        <div>✅ Split first, clean after — no information bleed</div>
        <div>✅ Encoders fit on training set only</div>
        <div>✅ SMOTE on training set only — never on test</div>
        <div>✅ Hyperparameters auto-tuned — not fixed manually</div>
        <div>✅ Primary metrics: AUC and F1 — not accuracy</div>
        <div>✅ Confusion matrix on test set predictions only</div>
        <div>✅ Aklan included — 17 records from new dataset</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================
# RISK ASSESSMENT
# ===========================================================================
elif page == "⚠️  Risk Assessment":
    st.markdown("<div class='page-title'>⚠️ At Risk of Completion Assessment</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Enter MSME details to assess project completion risk based on historical data.</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            province  = st.selectbox("Province", ["Aklan","Antique","Capiz","Guimaras","Iloilo","Negros"])
            sector    = st.selectbox("Sector", ["Food Processing","Furniture","Metals & Engineering",
                                                "Gifts, Decors, Handicrafts","Horticulture & Agriculture","Others (grouped)"])
        with c2:
            ownership = st.selectbox("Type of Ownership", ["Single","Corporation","Cooperative","Partnership"])
            size      = st.selectbox("Size of Enterprise", ["micro","small","medium"])
        with c3:
            cost      = st.number_input("Project Cost (₱)", min_value=10000, max_value=10000000, value=400000, step=10000)
            has_prior = st.radio("Has Prior Funding?", [False, True],
                                 format_func=lambda x: "Yes — 2nd or more project" if x else "No — 1st project")
        st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("🔍 Assess Completion Risk", use_container_width=True)

    if predict_btn:
        inp   = preprocess_input(art, province, sector, ownership, size, cost, has_prior)
        pred  = art['model'].predict(inp)[0]
        prob  = art['model'].predict_proba(inp)[0]
        p_c, p_nc = prob[1], prob[0]
        conf  = max(p_c, p_nc)

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1])

        with r1:
            if pred == 1:
                st.markdown(f"""
                <div class='result-complete'>
                    <div style='font-size:20px;font-weight:700;color:{C1}'>✅ LOW RISK — Likely to Complete</div>
                    <div style='font-size:13px;color:#444;margin-top:6px'>
                    Based on historical data, this MSME profile is predicted to
                    <b>complete</b> their project.
                    </div>
                    <div style='margin-top:12px;font-size:28px;font-weight:700;color:{C1}'>{p_c:.1%}</div>
                    <div style='font-size:12px;color:#6B8BBE'>Completion Probability</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-atrisk'>
                    <div style='font-size:20px;font-weight:700;color:{RED}'>⚠️ HIGH RISK — At Risk of Not Completing</div>
                    <div style='font-size:13px;color:#444;margin-top:6px'>
                    Based on historical data, this MSME profile is predicted to be
                    <b>at risk</b> of not completing their project.
                    </div>
                    <div style='margin-top:12px;font-size:28px;font-weight:700;color:{RED}'>{p_nc:.1%}</div>
                    <div style='font-size:12px;color:#E05A5A'>Non-Completion Probability</div>
                </div>
                """, unsafe_allow_html=True)

            conf_label = "High" if conf >= 0.75 else "Moderate" if conf >= 0.60 else "Low"
            conf_color = C1 if conf >= 0.75 else C2 if conf >= 0.60 else "#E09A2A"
            st.markdown(f"""
            <div style='font-size:13px;color:{conf_color};margin-top:4px'>
            📊 Model confidence: <b>{conf_label}</b> ({conf:.1%})
            {"" if conf >= 0.60 else " — interpret with caution"}
            </div>
            """, unsafe_allow_html=True)

        with r2:
            fig, ax = make_fig()
            fig.set_size_inches(4, 2.8)
            bars = ax.barh(['Not Completed','Completed'], [p_nc, p_c],
                           color=[RED, C2], height=0.45)
            for bar, val in zip(bars, [p_nc, p_c]):
                ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                        f'{val:.1%}', va='center', fontsize=11, fontweight='600', color='#333')
            ax.set_xlim(0, 1.2)
            ax.axvline(0.5, color='#CCC', linestyle='--', lw=1)
            ax.set_xlabel('Probability', fontsize=11)
            ax.set_title('Risk Probability', fontsize=12, fontweight='600')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:13px;font-weight:600;color:{C1};margin-bottom:6px'>Input Summary</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Province":province,"Sector":sector,"Ownership":ownership,
            "Size":size.capitalize(),"Project Cost":f"₱{cost:,.0f}",
            "Prior Funding":"Yes" if has_prior else "No",
            "Risk Assessment":"✅ Low Risk" if pred==1 else "⚠️ High Risk",
            "Probability":f"{p_c:.1%}" if pred==1 else f"{p_nc:.1%}"
        }]), use_container_width=True, hide_index=True)

# ===========================================================================
# MODEL PERFORMANCE
# ===========================================================================
elif page == "📈  Model Performance":
    st.markdown("<div class='page-title'>📈 Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Evaluated on held-out test set — 20% of data never seen during training.</div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ROC-AUC (Test)",     f"{art['test_auc']:.4f}")
    c2.metric("Weighted F1 (Test)", f"{art['test_f1']:.4f}")
    c3.metric("F1 (Not Completed)", f"{art['f1_nc']:.4f}")
    c4.metric("CV AUC (5-fold)",    f"{art['cv_auc']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<div style='font-size:15px;font-weight:600;color:{C1};margin-bottom:8px'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = confusion_matrix(art['y_test'], art['y_pred'])
        fig, ax = make_fig()
        fig.set_size_inches(4, 3.2)
        ConfusionMatrixDisplay(cm, display_labels=['Not Completed','Completed']).plot(
            ax=ax, cmap='Blues', colorbar=False)
        ax.set_title('Confusion Matrix — Test Set', fontsize=11, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close()

    with col2:
        st.markdown(f"<div style='font-size:15px;font-weight:600;color:{C1};margin-bottom:8px'>ROC Curve</div>", unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(art['y_test'], art['y_pred_proba'])
        fig, ax = make_fig()
        fig.set_size_inches(4, 3.2)
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
    st.markdown("<div class='page-sub'>Variables ranked by their contribution to the XGBoost model's predictions.</div>", unsafe_allow_html=True)

    imp = art['importance_df']
    col1, col2 = st.columns([3, 2])

    with col1:
        fig, ax = make_fig()
        fig.set_size_inches(5.5, 4)
        colors = [C1 if i < 3 else C3 for i in range(len(imp)-1,-1,-1)]
        bars   = ax.barh(imp['Feature'][::-1], imp['Importance'][::-1],
                         color=colors, height=0.55)
        for bar, val in zip(bars, imp['Importance'][::-1]):
            ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9, color='#333')
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.set_title('Feature Importance', fontsize=12, fontweight='600')
        top3   = mpatches.Patch(color=C1, label='Top 3')
        others = mpatches.Patch(color=C3, label='Others')
        ax.legend(handles=[top3,others], fontsize=9, framealpha=0.5)
        ax.set_xlim(0, imp['Importance'].max() * 1.25)
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
        <tr><td style='padding:4px 0;border-bottom:1px solid {C4}'><b>Province</b></td><td style='padding:4px 8px;border-bottom:1px solid {C4}'>Location of MSME</td></tr>
        <tr><td style='padding:4px 0;border-bottom:1px solid {C4}'><b>Sector</b></td><td style='padding:4px 8px;border-bottom:1px solid {C4}'>Industry sector</td></tr>
        <tr><td style='padding:4px 0;border-bottom:1px solid {C4}'><b>Type of Ownership</b></td><td style='padding:4px 8px;border-bottom:1px solid {C4}'>Single, Corp, Coop, Partnership</td></tr>
        <tr><td style='padding:4px 0;border-bottom:1px solid {C4}'><b>Size of Enterprise</b></td><td style='padding:4px 8px;border-bottom:1px solid {C4}'>Micro, Small, Medium</td></tr>
        <tr><td style='padding:4px 0;border-bottom:1px solid {C4}'><b>Project Cost</b></td><td style='padding:4px 8px;border-bottom:1px solid {C4}'>Total approved cost (₱)</td></tr>
        <tr><td style='padding:4px 0'><b>Has Prior Funding</b></td><td style='padding:4px 8px'>Previous DOST project</td></tr>
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
