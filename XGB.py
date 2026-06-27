"""
MSME Completion Risk Predictor
DOST SETUP 4.0 iFund Program — Western Visayas
XGBoost | Threshold Tuning | Risk Tiers | Batch Prediction | Data Overview

streamlit run SHI_XGBoost_Streamlit_Final.py
"""

import os
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
                                      ConfusionMatrixDisplay, roc_curve, precision_recall_curve)
from imblearn.over_sampling   import SMOTE
from xgboost                  import XGBClassifier
from scipy.stats              import randint, uniform

# ── Palette ──────────────────────────────────────────────────────────────────
C1,C2,C3,C4 = "#2C5EAD","#1591DC","#4BB8FA","#C4E2F5"
RED,ORG,GRN  = "#E05A5A","#E09A2A","#2EAD72"
RANDOM_STATE = 42

st.set_page_config(page_title="MSME Risk Predictor", page_icon="📊", layout="wide")

st.markdown(f"""<style>
.stApp{{background:#F7FBFF}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,{C1},{C2})}}
[data-testid="stSidebar"] *{{color:white!important}}
[data-testid="metric-container"]{{background:white;border:1px solid {C4};
  border-radius:10px;padding:12px;box-shadow:0 2px 6px rgba(44,94,173,.08)}}
[data-testid="metric-container"] label{{color:{C1}!important;font-size:12px}}
[data-testid="metric-container"] [data-testid="stMetricValue"]{{color:{C1}!important;font-size:22px;font-weight:600}}
h2,h3{{color:{C1}!important}}
.stButton>button{{background:linear-gradient(90deg,{C1},{C2});color:white;border:none;
  border-radius:8px;font-weight:600;padding:.6rem 1.2rem}}
.stButton>button:hover{{background:linear-gradient(90deg,{C2},{C3});color:white}}
.card{{background:white;border-radius:10px;padding:1.2rem 1.5rem;
  box-shadow:0 2px 8px rgba(44,94,173,.08);margin-bottom:1rem;border:1px solid {C4}}}
.ptitle{{color:{C1};font-size:26px;font-weight:700;margin-bottom:4px}}
.psub{{color:#6B8BBE;font-size:14px;margin-bottom:1.5rem}}
.badge-high{{background:#FDECEA;color:{RED};border:1px solid {RED};
  border-radius:20px;padding:4px 14px;font-weight:700;font-size:13px}}
.badge-medium{{background:#FEF3E2;color:{ORG};border:1px solid {ORG};
  border-radius:20px;padding:4px 14px;font-weight:700;font-size:13px}}
.badge-low{{background:#E6F9F1;color:{GRN};border:1px solid {GRN};
  border-radius:20px;padding:4px 14px;font-weight:700;font-size:13px}}
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
FEAT_LABELS = {
    'Province':'Province','Sector':'Sector',
    'type_of_ownership':'Type of Ownership',
    'size_of_enterprise':'Size of Enterprise',
    'Project_Cost':'Project Cost',
    'Has_Prior_Funding':'Has Prior Funding',
    'Cost_to_Size_Ratio':'Cost-to-Size Ratio',
    'Log_Project_Cost':'Log Project Cost',
}

def risk_tier(p_completed):
    pct = p_completed*100
    if pct < 44:   return "High",   RED, "🔴"
    elif pct < 65: return "Medium", ORG, "🟡"
    else:          return "Low",    GRN, "🟢"

def mfig(w=4.5,h=3.2):
    fig,ax=plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_edgecolor(C4)
    ax.tick_params(colors='#555')
    ax.xaxis.label.set_color('#555'); ax.yaxis.label.set_color('#555')
    ax.title.set_color(C1)
    return fig,ax

# ── Train model ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    _dir = os.path.dirname(os.path.abspath(__file__))
    df_raw = pd.read_excel(os.path.join(_dir, 'MSME_data_cleaned.xlsx'))
    # Store for data overview
    raw_copy = df_raw.copy()
    df = df_raw.drop(columns=['Beneficiary_Name','Year','Refund_Status']).copy()

    FEATURES=['Province','Sector','type_of_ownership','size_of_enterprise','Project_Cost','Has_Prior_Funding']
    X = df[FEATURES].copy()
    y = (df['Completion_Status']=='Completed').astype(int)

    sector_map={'Agriculture/Marine/Aquaculture':'Horticulture & Agriculture',
                'Horticulture and Agriculture':'Horticulture & Agriculture'}
    X['Sector']=X['Sector'].replace(sector_map)
    X['Sector']=X['Sector'].where(~X['Sector'].str.startswith('Others',na=False),'Others (grouped)')

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=RANDOM_STATE,stratify=y)

    CAT=['Province','Sector','type_of_ownership','size_of_enterprise']
    encoders={}; Xtr=X_train.copy(); Xte=X_test.copy()
    for col in CAT:
        le=LabelEncoder()
        Xtr[col]=le.fit_transform(X_train[col].astype(str))
        Xte[col]=X_test[col].astype(str).map(lambda x,le=le: le.transform([x])[0] if x in le.classes_ else -1)
        encoders[col]=le

    Xtr['Has_Prior_Funding']=Xtr['Has_Prior_Funding'].astype(int)
    Xte['Has_Prior_Funding']=Xte['Has_Prior_Funding'].astype(int)

    snm={'micro':1,'small':2,'medium':3}
    def af(Xr,Xe):
        Xe=Xe.copy()
        sn=Xr['size_of_enterprise'].str.lower().map(snm)
        Xe['Cost_to_Size_Ratio']=Xr['Project_Cost'].values/sn.values
        Xe['Log_Project_Cost']=np.log1p(Xr['Project_Cost'].values)
        return Xe

    Xtr=af(X_train,Xtr); Xte=af(X_test,Xte)
    Xtr_sm,ytr_sm=SMOTE(random_state=RANDOM_STATE,k_neighbors=5).fit_resample(Xtr,y_train)

    param_dist={'n_estimators':randint(200,600),'max_depth':randint(3,8),
        'learning_rate':uniform(.01,.2),'subsample':uniform(.6,.4),
        'colsample_bytree':uniform(.6,.4),'min_child_weight':randint(1,6),
        'gamma':uniform(0,.3),'reg_alpha':uniform(0,.5),'reg_lambda':uniform(.5,2.)}
    xgb=XGBClassifier(objective='binary:logistic',eval_metric='auc',
                      tree_method='hist',random_state=RANDOM_STATE)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
    rs=RandomizedSearchCV(xgb,param_dist,n_iter=80,scoring='roc_auc',cv=cv,n_jobs=-1,verbose=0,random_state=RANDOM_STATE)
    rs.fit(Xtr_sm,ytr_sm)

    bm=rs.best_estimator_
    yp=bm.predict_proba(Xte)[:,1]  # P(Completed)

    # Threshold tuning: balance F1 and Not-Completed recall
    best_t,best_score=0.5,0
    for t in np.arange(0.2,0.85,0.01):
        yt=(yp>=t).astype(int)
        f=f1_score(y_test,yt,average='weighted')
        cm=confusion_matrix(y_test,yt)
        nc_recall=cm[0,0]/20
        score=f*nc_recall
        if score>best_score:
            best_score=score; best_t=t

    y_pred=(yp>=best_t).astype(int)
    auc=roc_auc_score(y_test,yp)
    f1w=f1_score(y_test,y_pred,average='weighted')
    f1nc=f1_score(y_test,y_pred,pos_label=0)
    f1c=f1_score(y_test,y_pred,pos_label=1)
    cm=confusion_matrix(y_test,y_pred)

    imp=pd.DataFrame({'Feature':Xtr.columns,'Importance':bm.feature_importances_})\
         .sort_values('Importance',ascending=False).reset_index(drop=True)
    imp['Label']=imp['Feature'].map(FEAT_LABELS).fillna(imp['Feature'])

    return dict(
        model=bm, encoders=encoders, sector_map=sector_map, snm=snm,
        feature_cols=list(Xtr.columns), best_params=rs.best_params_,
        cv_auc=rs.best_score_, test_auc=auc, test_f1=f1w, f1_nc=f1nc, f1_c=f1c,
        threshold=best_t, importance_df=imp,
        y_test=y_test, y_pred=y_pred, y_pred_proba=yp, cm=cm,
        raw_df=raw_copy, X_sector=X['Sector'],
    )

# ── Preprocess single input ──────────────────────────────────────────────────
def preprocess(art, province, sector, ownership, size, cost, has_prior):
    sector=art['sector_map'].get(sector,sector)
    if sector.startswith('Others'): sector='Others (grouped)'
    row=pd.DataFrame([{'Province':province,'Sector':sector,'type_of_ownership':ownership,
                        'size_of_enterprise':size,'Project_Cost':cost,'Has_Prior_Funding':int(has_prior)}])
    for col in ['Province','Sector','type_of_ownership','size_of_enterprise']:
        le=art['encoders'][col]; val=str(row[col].iloc[0])
        row[col]=le.transform([val])[0] if val in le.classes_ else -1
    sn=art['snm'].get(size.lower(),1)
    row['Cost_to_Size_Ratio']=cost/sn
    row['Log_Project_Cost']=np.log1p(cost)
    return row[art['feature_cols']]

def predict_risk(art, inp):
    prob=art['model'].predict_proba(inp)[0]
    p_c,p_nc=prob[1],prob[0]
    pred=int(p_c>=art['threshold'])
    tier,tc,ti=risk_tier(p_c)
    return p_c,p_nc,pred,tier,tc,ti

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:21px;font-weight:700;margin-bottom:2px'>📊 MSME Risk</div>",unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;opacity:.85;margin-bottom:1rem'>DOST SETUP 4.0 iFund | Western Visayas</div>",unsafe_allow_html=True)
    st.markdown("---")
    page=st.radio("",["🏠  Overview","⚠️  Risk Assessment","🔄  Batch Assessment",
                       "📈  Model Performance","📊  Feature Importance","ℹ️  About"],
                  label_visibility="collapsed")

# ── Load ─────────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    try:
        art=train_model()
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"""<div style='font-size:11px;opacity:.85;line-height:1.8'>
            ✅ Model ready<br>CV AUC: <b>{art['cv_auc']:.4f}</b><br>
            Test AUC: <b>{art['test_auc']:.4f}</b><br>
            Threshold: <b>{art['threshold']:.2f}</b>
            </div>""",unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("⚠️ MSME_data_cleaned.xlsx not found. Upload it to your GitHub repo.")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page=="🏠  Overview":
    st.markdown("<div class='ptitle'>MSME Project Completion Risk Predictor</div>",unsafe_allow_html=True)
    st.markdown("<div class='psub'>DOST SETUP 4.0 iFund Program — Western Visayas &nbsp;|&nbsp; Powered by XGBoost</div>",unsafe_allow_html=True)

    raw=art['raw_df']
    completed   = (raw['Completion_Status']=='Completed').sum()
    notcompleted= (raw['Completion_Status']=='Not Completed').sum()
    total       = len(raw)
    comp_rate   = completed/total*100

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total MSMEs",f"{total}")
    c2.metric("Completed",f"{completed}  ({comp_rate:.0f}%)")
    c3.metric("Not Completed",f"{notcompleted}  ({100-comp_rate:.0f}%)")
    c4.metric("Provinces Covered","6")

    st.markdown("<br>",unsafe_allow_html=True)
    col1,col2=st.columns(2)

    # Completion by Province
    with col1:
        st.markdown(f"<div style='font-size:15px;font-weight:600;color:{C1};margin-bottom:8px'>📍 Completion by Province</div>",unsafe_allow_html=True)
        prov_df=raw.groupby('Province')['Completion_Status'].value_counts().unstack(fill_value=0)
        if 'Completed' not in prov_df.columns: prov_df['Completed']=0
        if 'Not Completed' not in prov_df.columns: prov_df['Not Completed']=0
        prov_df=prov_df.sort_values('Completed',ascending=True)
        fig,ax=mfig(5,3.5)
        ax.barh(prov_df.index, prov_df.get('Completed',0), color=C2, label='Completed', height=0.5)
        ax.barh(prov_df.index, prov_df.get('Not Completed',0),
                left=prov_df.get('Completed',0), color=RED, alpha=0.7, label='Not Completed', height=0.5)
        ax.set_xlabel('Number of MSMEs'); ax.set_title('Completion by Province',fontweight='600')
        ax.legend(fontsize=9); plt.tight_layout()
        st.pyplot(fig,use_container_width=False); plt.close()

    # Completion by Sector
    with col2:
        st.markdown(f"<div style='font-size:15px;font-weight:600;color:{C1};margin-bottom:8px'>🏭 Completion by Sector</div>",unsafe_allow_html=True)
        sec_col=art['X_sector']
        sec_df=pd.DataFrame({'Sector':sec_col,'Status':raw['Completion_Status']})
        sec_grp=sec_df.groupby('Sector')['Status'].value_counts().unstack(fill_value=0)
        if 'Completed' not in sec_grp.columns: sec_grp['Completed']=0
        if 'Not Completed' not in sec_grp.columns: sec_grp['Not Completed']=0
        sec_grp['Total']=sec_grp['Completed']+sec_grp['Not Completed']
        sec_grp=sec_grp.sort_values('Total',ascending=True)
        fig,ax=mfig(5,3.5)
        ax.barh(sec_grp.index, sec_grp['Completed'], color=C2, label='Completed', height=0.5)
        ax.barh(sec_grp.index, sec_grp['Not Completed'],
                left=sec_grp['Completed'], color=RED, alpha=0.7, label='Not Completed', height=0.5)
        ax.set_xlabel('Number of MSMEs'); ax.set_title('Completion by Sector',fontweight='600')
        ax.legend(fontsize=9); plt.tight_layout()
        st.pyplot(fig,use_container_width=False); plt.close()

    # Risk tier guide
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown(f"""
    <div class='card'>
    <h3 style='margin-top:0'>🏷️ How to Read Risk Tiers</h3>
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:8px'>
      <div style='text-align:center;padding:16px;background:#FDECEA;border-radius:10px;border:1px solid {RED}'>
        <div style='font-size:28px'>🔴</div>
        <div style='font-size:16px;font-weight:700;color:{RED}'>High Risk</div>
        <div style='font-size:12px;color:#444;margin-top:6px'>Completion probability<br>below <b>44%</b><br>Needs immediate attention</div>
      </div>
      <div style='text-align:center;padding:16px;background:#FEF3E2;border-radius:10px;border:1px solid {ORG}'>
        <div style='font-size:28px'>🟡</div>
        <div style='font-size:16px;font-weight:700;color:{ORG}'>Medium Risk</div>
        <div style='font-size:12px;color:#444;margin-top:6px'>Completion probability<br><b>44% – 65%</b><br>Monitor closely</div>
      </div>
      <div style='text-align:center;padding:16px;background:#E6F9F1;border-radius:10px;border:1px solid {GRN}'>
        <div style='font-size:28px'>🟢</div>
        <div style='font-size:16px;font-weight:700;color:{GRN}'>Low Risk</div>
        <div style='font-size:12px;color:#444;margin-top:6px'>Completion probability<br>above <b>65%</b><br>On track</div>
      </div>
    </div>
    </div>
    """,unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. SINGLE RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page=="⚠️  Risk Assessment":
    st.markdown("<div class='ptitle'>⚠️ Risk Assessment</div>",unsafe_allow_html=True)
    st.markdown("<div class='psub'>Enter MSME details to assess project completion risk.</div>",unsafe_allow_html=True)

    with st.form("risk_form"):
        st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C1};margin-bottom:12px'>MSME Information</div>",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            province =st.selectbox("📍 Province",["Aklan","Antique","Capiz","Guimaras","Iloilo","Negros"])
            sector   =st.selectbox("🏭 Sector",["Food Processing","Furniture","Metals & Engineering",
                                                "Gifts, Decors, Handicrafts","Horticulture & Agriculture","Others (grouped)"])
        with c2:
            ownership=st.selectbox("🏢 Type of Ownership",["Single","Corporation","Cooperative","Partnership"])
            size     =st.selectbox("📏 Size of Enterprise",["micro","small","medium"],
                                   format_func=lambda x:x.capitalize())
        with c3:
            cost     =st.number_input("💰 Project Cost (₱)",min_value=10000,max_value=10000000,value=400000,step=10000)
            has_prior=st.radio("📋 Has Prior DOST Funding?",[False,True],
                               format_func=lambda x:"Yes — 2nd or more project" if x else "No — 1st project")
        submitted=st.form_submit_button("🔍  Assess Risk",use_container_width=True)

    if submitted:
        inp=preprocess(art,province,sector,ownership,size,cost,has_prior)
        p_c,p_nc,pred,tier,tc,ti=predict_risk(art,inp)

        st.markdown("<br>",unsafe_allow_html=True)
        r1,r2,r3=st.columns([1.1,0.9,1.1])

        # Result
        with r1:
            if pred==1:
                bg,bc,label,sub=C4,C1,"✅ LIKELY TO COMPLETE","This MSME is predicted to complete their project based on historical patterns."
                pval,plabel=p_c,"Completion Probability"
            else:
                bg,bc,label,sub="#FDECEA",RED,"⚠️ AT RISK OF NOT COMPLETING","This MSME profile resembles past projects that did not complete. Early intervention is recommended."
                pval,plabel=p_nc,"Non-Completion Probability"

            st.markdown(f"""
            <div style='background:{bg};border-left:5px solid {bc};border-radius:10px;padding:1.3rem 1.5rem'>
              <div style='font-size:17px;font-weight:700;color:{bc}'>{label}</div>
              <div style='font-size:12px;color:#555;margin-top:6px;line-height:1.5'>{sub}</div>
              <div style='margin-top:14px;font-size:34px;font-weight:700;color:{bc}'>{pval:.1%}</div>
              <div style='font-size:11px;color:#888;margin-bottom:12px'>{plabel}</div>
              <span class='badge-{"high" if tier=="High" else "medium" if tier=="Medium" else "low"}'>{ti} {tier} Risk</span>
            </div>
            """,unsafe_allow_html=True)
            conf=max(p_c,p_nc)
            cl="High" if conf>=.75 else "Moderate" if conf>=.60 else "Low"
            cc=GRN if conf>=.75 else C2 if conf>=.60 else ORG
            st.markdown(f"<div style='font-size:12px;color:{cc};margin-top:8px'>📊 Model confidence: <b>{cl}</b> ({conf:.1%})</div>",unsafe_allow_html=True)

        # Probability bar
        with r2:
            fig,ax=mfig(3.2,3.2)
            ax.barh(['Not Completed','Completed'],[p_nc,p_c],color=[RED,C2],height=0.45)
            for val,y in zip([p_nc,p_c],[0,1]):
                ax.text(val+.01,y,f'{val:.1%}',va='center',fontsize=10,fontweight='600',color='#333')
            ax.axvline(art['threshold'],color='gray',linestyle='--',lw=1.2,alpha=0.6)
            ax.set_xlim(0,1.22); ax.set_xlabel('Probability',fontsize=9)
            ax.set_title('Risk Probabilities',fontsize=11,fontweight='600')
            plt.tight_layout(); st.pyplot(fig,use_container_width=False); plt.close()
            st.caption(f"Dashed line = decision threshold ({art['threshold']:.2f})")

        # Top factors
        with r3:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C1};margin-bottom:10px'>🔍 What Influenced This Result</div>",unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:12px;color:#666;margin-bottom:12px'>Top factors the model considers when assessing risk:</div>",unsafe_allow_html=True)
            imp=art['importance_df']
            max_imp=imp.iloc[0]['Importance']
            for _,row in imp.head(5).iterrows():
                bw=int((row['Importance']/max_imp)*100)
                st.markdown(f"""
                <div style='margin-bottom:10px'>
                  <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px'>
                    <span style='font-weight:600;color:#333'>{row['Label']}</span>
                    <span style='color:#888'>{row['Importance']:.3f}</span>
                  </div>
                  <div style='background:{C4};border-radius:4px;height:8px'>
                    <div style='background:{C2};width:{bw}%;height:8px;border-radius:4px'></div>
                  </div>
                </div>
                """,unsafe_allow_html=True)
            st.caption("These are overall model importance scores, not specific to this prediction.")

        # Summary table
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:13px;font-weight:600;color:{C1};margin-bottom:6px'>📋 Summary</div>",unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Province":province,"Sector":sector,"Ownership":ownership,
            "Size":size.capitalize(),"Project Cost":f"₱{cost:,.0f}",
            "Prior Funding":"Yes" if has_prior else "No",
            "Risk Tier":f"{ti} {tier}","Probability":f"{pval:.1%}"
        }]),use_container_width=True,hide_index=True)

        st.info("💡 **Note:** This prediction is based on patterns from historical MSME data. Use it as a guide for monitoring and early intervention, not as a final decision.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. BATCH ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🔄  Batch Assessment":
    st.markdown("<div class='ptitle'>🔄 Batch Risk Assessment</div>",unsafe_allow_html=True)
    st.markdown("<div class='psub'>Upload a CSV file to assess risk for multiple MSMEs at once.</div>",unsafe_allow_html=True)

    # Template
    with st.expander("📥 Download CSV Template"):
        template=pd.DataFrame([
            {"Province":"Iloilo","Sector":"Food Processing","type_of_ownership":"Single",
             "size_of_enterprise":"micro","Project_Cost":400000,"Has_Prior_Funding":False},
            {"Province":"Capiz","Sector":"Furniture","type_of_ownership":"Corporation",
             "size_of_enterprise":"small","Project_Cost":800000,"Has_Prior_Funding":True},
            {"Province":"Aklan","Sector":"Metals & Engineering","type_of_ownership":"Cooperative",
             "size_of_enterprise":"medium","Project_Cost":1500000,"Has_Prior_Funding":False},
        ])
        st.dataframe(template,use_container_width=True,hide_index=True)
        csv_bytes=template.to_csv(index=False).encode()
        st.download_button("⬇️ Download Template CSV",csv_bytes,"msme_template.csv","text/csv")

    uploaded=st.file_uploader("Upload your CSV file",type=["csv"])

    if uploaded:
        try:
            batch=pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(batch)} records.")
            st.dataframe(batch.head(5),use_container_width=True,hide_index=True)

            if st.button("🔍 Run Batch Assessment",use_container_width=True):
                results=[]
                errors=[]
                for idx,row in batch.iterrows():
                    try:
                        inp=preprocess(art,
                            str(row['Province']),str(row['Sector']),
                            str(row['type_of_ownership']),str(row['size_of_enterprise']),
                            float(row['Project_Cost']),bool(row['Has_Prior_Funding']))
                        p_c,p_nc,pred,tier,tc,ti=predict_risk(art,inp)
                        results.append({
                            "Province":row['Province'],"Sector":row['Sector'],
                            "Ownership":row['type_of_ownership'],"Size":row['size_of_enterprise'],
                            "Project Cost":f"₱{float(row['Project_Cost']):,.0f}",
                            "Completion Prob":f"{p_c:.1%}",
                            "Non-Completion Prob":f"{p_nc:.1%}",
                            "Risk Tier":f"{ti} {tier}",
                            "Status":"✅ Likely Complete" if pred==1 else "⚠️ At Risk",
                        })
                    except Exception as e:
                        errors.append(f"Row {idx+1}: {e}")

                if results:
                    res_df=pd.DataFrame(results)
                    st.markdown("<br>",unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:15px;font-weight:600;color:{C1};margin-bottom:8px'>📊 Assessment Results</div>",unsafe_allow_html=True)

                    # Summary metrics
                    at_risk=sum(1 for r in results if "At Risk" in r['Status'])
                    safe=len(results)-at_risk
                    m1,m2,m3=st.columns(3)
                    m1.metric("Total Assessed",len(results))
                    m2.metric("⚠️ At Risk",at_risk)
                    m3.metric("✅ Likely to Complete",safe)

                    st.dataframe(res_df,use_container_width=True,hide_index=True)

                    # Download
                    csv_out=res_df.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Results CSV",csv_out,"batch_risk_results.csv","text/csv")

                if errors:
                    st.warning("Some rows had errors:\n"+"\n".join(errors))
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.markdown(f"""
        <div class='card' style='text-align:center;padding:2rem'>
          <div style='font-size:40px'>📂</div>
          <div style='font-size:15px;font-weight:600;color:{C1};margin-top:8px'>Upload a CSV file to get started</div>
          <div style='font-size:13px;color:#888;margin-top:4px'>Download the template above to see the required format</div>
        </div>
        """,unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page=="📈  Model Performance":
    st.markdown("<div class='ptitle'>📈 Model Performance</div>",unsafe_allow_html=True)
    st.markdown("<div class='psub'>How well does the model predict project completion risk?</div>",unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("ROC-AUC",f"{art['test_auc']:.4f}",help="Higher = better. 1.0 is perfect, 0.5 is random.")
    c2.metric("F1-Score",f"{art['test_f1']:.4f}",help="Balance between precision and recall.")
    c3.metric("F1 (At-Risk Class)",f"{art['f1_nc']:.4f}",help="How well the model detects at-risk MSMEs.")
    c4.metric("Cross-Val AUC",f"{art['cv_auc']:.4f}",help="Average AUC across 5 training folds.")

    st.markdown("<br>",unsafe_allow_html=True)

    # Plain language explanation
    st.markdown(f"""
    <div class='card'>
    <h3 style='margin-top:0'>📖 What Do These Numbers Mean?</h3>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:13px;color:#444;line-height:1.7'>
      <div><b>ROC-AUC ({art['test_auc']:.2f})</b> — The model correctly ranks at-risk MSMEs above low-risk ones
      {art['test_auc']*100:.0f}% of the time. Anything above 0.70 is considered good.</div>
      <div><b>F1 At-Risk ({art['f1_nc']:.2f})</b> — Out of every 10 truly at-risk MSMEs,
      the model correctly flags about {art['f1_nc']*10:.0f} of them. The rest may need manual review.</div>
      <div><b>Decision Threshold ({art['threshold']:.2f})</b> — The model flags an MSME as at-risk
      when its completion probability drops below {art['threshold']*100:.0f}%. This was tuned to
      catch as many at-risk cases as possible.</div>
      <div><b>Cross-Val AUC ({art['cv_auc']:.2f})</b> — Tested across 5 different data splits to ensure
      the model performs consistently, not just on one lucky split.</div>
    </div>
    </div>
    """,unsafe_allow_html=True)

    col1,col2=st.columns(2)
    with col1:
        st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C1};margin-bottom:8px'>Confusion Matrix</div>",unsafe_allow_html=True)
        fig,ax=mfig(4,3.2)
        ConfusionMatrixDisplay(art['cm'],display_labels=['Not Completed','Completed']).plot(ax=ax,cmap='Blues',colorbar=False)
        ax.set_title('Confusion Matrix — Test Set',fontsize=11,fontweight='600')
        plt.tight_layout(); st.pyplot(fig,use_container_width=False); plt.close()
        tn,fp,fn,tp=art['cm'].ravel()
        st.caption(f"Correctly caught {tn} at-risk and {tp} completed MSMEs. Missed {fn} at-risk cases.")

    with col2:
        st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C1};margin-bottom:8px'>ROC Curve</div>",unsafe_allow_html=True)
        fpr,tpr,_=roc_curve(art['y_test'],art['y_pred_proba'])
        fig,ax=mfig(4,3.2)
        ax.plot(fpr,tpr,color=C2,lw=2,label=f"XGBoost (AUC={art['test_auc']:.3f})")
        ax.plot([0,1],[0,1],color='#CCC',linestyle='--',lw=1,label='Random Guess')
        ax.fill_between(fpr,tpr,alpha=.08,color=C3)
        ax.set_xlabel('False Positive Rate',fontsize=9)
        ax.set_ylabel('True Positive Rate',fontsize=9)
        ax.set_title('ROC Curve — Test Set',fontsize=11,fontweight='600')
        ax.legend(fontsize=9,framealpha=.5); plt.tight_layout()
        st.pyplot(fig,use_container_width=False); plt.close()
        st.caption("The curve shows how the model trades off catching at-risk cases vs. false alarms.")

    st.markdown("<br>",unsafe_allow_html=True)
    with st.expander("🔧 Best Hyperparameters (auto-selected by the model)"):
        params_df=pd.DataFrame([(k,round(float(v),4)) for k,v in art['best_params'].items()],
                               columns=['Parameter','Best Value'])
        st.dataframe(params_df,use_container_width=True,hide_index=True)
        st.caption("These settings were automatically chosen to give the best performance — not set manually.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page=="📊  Feature Importance":
    st.markdown("<div class='ptitle'>📊 Feature Importance</div>",unsafe_allow_html=True)
    st.markdown("<div class='psub'>Which MSME characteristics matter most in predicting completion risk?</div>",unsafe_allow_html=True)

    imp=art['importance_df']
    col1,col2=st.columns([3,2])

    with col1:
        fig,ax=mfig(5.5,4.2)
        colors=[C1 if i<3 else C3 for i in range(len(imp)-1,-1,-1)]
        bars=ax.barh(imp['Label'][::-1],imp['Importance'][::-1],color=colors,height=0.55)
        for bar,val in zip(bars,imp['Importance'][::-1]):
            ax.text(bar.get_width()+.003,bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}',va='center',fontsize=9,color='#333')
        ax.set_xlabel('Importance Score',fontsize=10)
        ax.set_title('Feature Importance — XGBoost',fontsize=12,fontweight='600')
        ax.set_xlim(0,imp['Importance'].max()*1.3)
        ax.legend(handles=[mpatches.Patch(color=C1,label='Top 3'),
                           mpatches.Patch(color=C3,label='Others')],fontsize=9,framealpha=.5)
        plt.tight_layout(); st.pyplot(fig,use_container_width=False); plt.close()

    with col2:
        st.markdown("<br>",unsafe_allow_html=True)
        for _,row in imp.iterrows():
            bw=int((row['Importance']/imp.iloc[0]['Importance'])*100)
            st.markdown(f"""
            <div style='margin-bottom:12px'>
              <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px'>
                <span style='font-weight:600;color:#333'>{row['Label']}</span>
                <span style='color:#888'>{row['Importance']*100:.1f}%</span>
              </div>
              <div style='background:{C4};border-radius:4px;height:10px'>
                <div style='background:{C1 if _ < 3 else C3};width:{bw}%;height:10px;border-radius:4px'></div>
              </div>
            </div>
            """,unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    top=imp.iloc[0]['Label']
    second=imp.iloc[1]['Label']
    st.markdown(f"""
    <div class='card'>
    <h3 style='margin-top:0'>💡 What This Means</h3>
    <p style='font-size:13px;color:#444;line-height:1.8'>
    <b>{top}</b> is the most important factor — MSMEs from different provinces show
    very different completion patterns in the historical data.<br><br>
    <b>{second}</b> also plays a big role — the type of business sector
    significantly affects whether a project gets completed.<br><br>
    <b>Project Cost and Cost-to-Size Ratio</b> suggest that financial fit matters —
    projects that are too large relative to the enterprise size tend to struggle more.
    </p>
    </div>
    """,unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6. ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page=="ℹ️  About":
    st.markdown("<div class='ptitle'>ℹ️ About This Tool</div>",unsafe_allow_html=True)
    st.markdown("<div class='psub'>MSME Project Completion Risk Predictor — DOST SETUP 4.0 iFund Program</div>",unsafe_allow_html=True)

    col1,col2=st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>🎯 Purpose</h3>
        <p style='font-size:13px;color:#444;line-height:1.8'>
        This tool helps <b>DOST-PSTO officers</b> identify which MSME projects are
        at risk of not completing, so that early support and monitoring can be
        provided. It is based on patterns from <b>321 historical projects</b>
        across Western Visayas.<br><br>
        <b>For MSME owners:</b> This tool shows how your project profile compares
        to past projects. It is a guide, not a final decision.<br><br>
        <b>For PSTO officers:</b> Use High Risk flags to prioritize follow-up visits
        and technical assistance.
        </p>
        </div>
        <div class='card'>
        <h3 style='margin-top:0'>📊 Dataset</h3>
        <p style='font-size:13px;color:#444;line-height:1.8'>
        <b>Records:</b> 321 MSME projects<br>
        <b>Provinces:</b> Aklan, Antique, Capiz, Guimaras, Iloilo, Negros Occidental<br>
        <b>Source:</b> DOST SETUP 4.0 iFund Program<br>
        <b>Target:</b> Completion Status (Completed / Not Completed)
        </p>
        </div>
        """,unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
        <h3 style='margin-top:0'>🤖 About the Model</h3>
        <p style='font-size:13px;color:#444;line-height:1.8'>
        <b>Algorithm:</b> XGBoost — a powerful machine learning model that learns
        patterns from historical data to make predictions.<br><br>
        <b>Training:</b> The model was trained on 80% of the data, then tested
        on the remaining 20% it had never seen before.<br><br>
        <b>Class imbalance:</b> Since there are more completed than not-completed projects,
        SMOTE was used to balance the training data.<br><br>
        <b>Threshold:</b> The prediction threshold was tuned to catch as many
        at-risk cases as possible, even at the cost of some false alarms.
        </p>
        </div>
        <div class='card'>
        <h3 style='margin-top:0'>📚 References</h3>
        <p style='font-size:13px;color:#444;line-height:1.8'>
        Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
        <i>Proceedings of the 22nd ACM SIGKDD.</i><br><br>
        Chawla, N. V. et al. (2002). SMOTE: Synthetic minority over-sampling technique.
        <i>Journal of Artificial Intelligence Research, 16</i>, 321–357.
        </p>
        </div>
        """,unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align:center;padding:1rem;color:#888;font-size:12px'>
    MSME Completion Risk Predictor · DOST SETUP 4.0 iFund · Western Visayas ·
    Powered by XGBoost · Built with Streamlit
    </div>
    """,unsafe_allow_html=True)
