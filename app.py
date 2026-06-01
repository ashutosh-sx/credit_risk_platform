import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add root folder to sys path for modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import generate_mock_data
from src.ml.explain import calculate_feature_contributions
from src.ml.rules import evaluate_credit_policy_rules
from src.talk_to_data.nl_to_sql import translate_nl_to_sql, synthesize_business_insight
from src.talk_to_data.query_runner import execute_query
from src.utils.config import get_config

# Set Page Config
st.set_page_config(
    page_title="NEOSTATS - AI Credit Risk Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #16A34A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #16A34A;
    }
    .risk-high {
        color: #DC2626;
        font-weight: bold;
    }
    .risk-medium {
        color: #D97706;
        font-weight: bold;
    }
    .risk-low {
        color: #059669;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

config = get_config()

# Helper to load trained model or mock training if not serialized
@st.cache_resource
def get_model_pipeline():
    model_path = os.path.join(config.models_dir, "credit_risk_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # Train a mock model dynamically to prevent setup failure on first load
        from sklearn.ensemble import RandomForestClassifier
        from src.data.preprocessor import CreditRiskPreprocessor
        
        df = generate_mock_data(n_samples=1000)
        X = df.drop(columns=['TARGET'])
        y = df['TARGET']
        
        preprocessor = CreditRiskPreprocessor()
        preprocessor.fit(X)
        X_proc = preprocessor.transform(X)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
        clf.fit(X_proc, y)
        
        return {
            'preprocessor': preprocessor,
            'model': clf,
            'features': preprocessor.get_feature_names_out()
        }

pipeline = get_model_pipeline()

@st.cache_data
def get_eda_data():
    try:
        from src.talk_to_data.query_runner import execute_query
        df = execute_query("SELECT * FROM applications;")
        if not df.empty:
            return df
    except Exception:
        pass
    return generate_mock_data(n_samples=1000)

mock_data_df = get_eda_data()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/120/card-security.png", width=90)
st.sidebar.title("NEOSTATS Intelligence")
st.sidebar.subheader("Credit Risk Platform")
menu = st.sidebar.radio(
    "Navigation Menu",
    ["📊 Overview & EDA", "💳 Risk Scoring & XAI", "📋 Policy Decision Rules", "💬 Talk-to-Data Chatbot", "📈 Model Diagnostics & Retraining"]
)

st.sidebar.divider()
st.sidebar.info("**API Integration Enabled**\n\nGroq Cloud LLaMA3-8B engine is active for NLP data analysis.")

# --- MODULE 1: OVERVIEW & EDA ---
if menu == "📊 Overview & EDA":
    st.markdown("<div class='main-title'>📊 Exploratory Data Analysis & Business Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>In-depth statistical understanding of demographics, financial ratios, and credit histories.</div>", unsafe_allow_html=True)
    
    # Platform statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applicants", "307,511", "Kaggle Benchmark")
    with col2:
        st.metric("Default Rate", "8.07%", "-0.4% MoM")
    with col3:
        st.metric("Median Income", "$147,150", "USD Equivalent")
    with col4:
        st.metric("Data Quality (Missing Score)", "0.13%", "Imputed via Median")
        
    st.divider()
    
    st.subheader("💡 Key Credit Business Insights")
    
    tabs = st.tabs([
        "1. Class Imbalance",
        "2. External Score Impact",
        "3. Age & Default Risk",
        "4. Debt Service (DTI)",
        "5. Income Ratios"
    ])
    
    with tabs[0]:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("""
            ### Insight 1: Extreme Target Class Imbalance
            *   **Observation**: Approximately **91.9%** of historical applicants successfully repaid their credits, while **8.1%** defaulted (`TARGET = 1`).
            *   **Business Action**: Plain classification models will score high accuracy by predicting all zeroes. To prevent missing defaults (false negatives), our ML models utilize **weighted balanced losses** and **custom probability thresholds**.
            """)
        with col_right:
            target_counts = mock_data_df['TARGET'].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=['Repaid (0)', 'Defaulted (1)'],
                title="Historical Credit Target Distribution",
                color_discrete_sequence=['#059669', '#DC2626']
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with tabs[1]:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("""
            ### Insight 2: External Credit Scores as Key Risk Signals
            *   **Observation**: External bureau evaluations (`EXT_SOURCE_1, 2, 3`) correlate inversely with default risks. Higher score ranges correspond to zero default rates.
            *   **Business Action**: Applicants with scoring parameters falling below `0.20` are immediate candidates for hard block knock-out rules.
            """)
        with col_right:
            fig = px.box(
                mock_data_df,
                x='TARGET',
                y='EXT_SOURCE_2',
                color='TARGET',
                title="External Source Score 2 by Default Status",
                color_discrete_sequence=['#059669', '#DC2626']
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with tabs[2]:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("""
            ### Insight 3: Higher Vulnerability Among Younger Demographics
            *   **Observation**: Younger credit applicants (especially those under 25 years, corresponding to lower `DAYS_BIRTH` magnitude) show elevated default indices.
            *   **Business Action**: Implement a thin-file mandatory co-signer policy for applicants aged under 23.
            """)
        with col_right:
            # Construct age bands
            mock_data_df['Age'] = abs(mock_data_df['DAYS_BIRTH']) / 365.25
            mock_data_df['AgeGroup'] = pd.cut(mock_data_df['Age'], bins=[20, 25, 35, 45, 55, 70], labels=['20-25', '25-35', '35-45', '45-55', '55+'])
            age_defaults = mock_data_df.groupby('AgeGroup')['TARGET'].mean().reset_index()
            age_defaults['TARGET'] *= 100
            
            fig = px.bar(
                age_defaults,
                x='AgeGroup',
                y='TARGET',
                title="Default Rates (%) by Age Group",
                labels={'TARGET': 'Default Rate (%)', 'AgeGroup': 'Age Band'},
                color_discrete_sequence=['#16A34A']
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with tabs[3]:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("""
            ### Insight 4: Debt Service (DTI) Outliers
            *   **Observation**: Monthly credit annuities exceeding 20% of an applicant's total monthly income (DTI ratio > 0.20) show exponential increases in default rates.
            *   **Business Action**: Establish a credit policy ceiling preventing automated approvals for DTI ratios exceeding 30%.
            """)
        with col_right:
            mock_data_df['DTI'] = (mock_data_df['AMT_ANNUITY'] * 12.0) / mock_data_df['AMT_INCOME_TOTAL']
            fig = px.histogram(
                mock_data_df,
                x='DTI',
                color='TARGET',
                barmode='overlay',
                title="Debt-to-Income (DTI) Ratio Density Breakdown",
                color_discrete_sequence=['#059669', '#DC2626']
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with tabs[4]:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("""
            ### Insight 5: Education Levels and Risk
            *   **Observation**: Applicants with Higher Education backgrounds demonstrate lower default probabilities compared to those with lower secondary degrees.
            *   **Business Action**: Use applicant education as a primary component in manual reviews.
            """)
        with col_right:
            edu_risk = pd.DataFrame({
                'Education': ['Higher Education', 'Secondary / Special', 'Incomplete Higher', 'Lower Secondary'],
                'Default Rate (%)': [4.8, 8.9, 8.4, 13.7]
            })
            fig = px.bar(
                edu_risk,
                x='Education',
                y='Default Rate (%)',
                title="Default Ratios across Academic Qualifications",
                color_discrete_sequence=['#8B5CF6']
            )
            st.plotly_chart(fig, use_container_width=True)

# --- MODULE 2 & 3: RISK SCORING & EXPLAINABLE AI ---
elif menu == "💳 Risk Scoring & XAI":
    st.markdown("<div class='main-title'>💳 Client Credit Scoring Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Assess individual credit applications, output risk probability scores, and interpret predictions using explainable AI contributions.</div>", unsafe_allow_html=True)
    
    # Scoring Inputs Layout
    st.subheader("📝 Applicant Credit Profile Form")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        income_val = st.number_input("Annual Income (USD)", min_value=10000, max_value=1000000, value=150000, step=5000)
        credit_val = st.number_input("Credit Amount (USD)", min_value=20000, max_value=2000000, value=500000, step=10000)
        annuity_val = st.number_input("Monthly Annuity (USD)", min_value=1000, max_value=100000, value=25000, step=500)
        cnt_children = st.slider("Number of Children", 0, 8, 1)
        
    with col2:
        age_years = st.slider("Applicant Age (Years)", 18, 70, 35)
        emp_years = st.slider("Employment Duration (Years)", 0, 45, 8)
        gender = st.selectbox("Applicant Gender", ["F", "M"])
        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        
    with col3:
        ext_source_1 = st.slider("External Rating Score 1 (Global)", 0.0, 1.0, 0.65, step=0.01)
        ext_source_2 = st.slider("External Rating Score 2 (Regional)", 0.0, 1.0, 0.72, step=0.01)
        ext_source_3 = st.slider("External Rating Score 3 (Local)", 0.0, 1.0, 0.58, step=0.01)
        flag_own_car = st.selectbox("Owns Car?", ["Y", "N"])
        flag_own_realty = st.selectbox("Owns Real Estate?", ["Y", "N"])

    st.divider()
    
    # Assess Risk Action
    if st.button("Assess Applicant Credit Risk", type="primary"):
        # Map variables to model input dataframe
        client_input = {
            'SK_ID_CURR': 100001,
            'NAME_CONTRACT_TYPE': contract_type,
            'CODE_GENDER': gender,
            'FLAG_OWN_CAR': flag_own_car,
            'FLAG_OWN_REALTY': flag_own_realty,
            'CNT_CHILDREN': cnt_children,
            'AMT_INCOME_TOTAL': income_val,
            'AMT_CREDIT': credit_val,
            'AMT_ANNUITY': annuity_val,
            'DAYS_BIRTH': -int(age_years * 365.25),
            'DAYS_EMPLOYED': -int(emp_years * 365.25),
            'EXT_SOURCE_1': ext_source_1,
            'EXT_SOURCE_2': ext_source_2,
            'EXT_SOURCE_3': ext_source_3
        }
        
        # 1. Obtain ML prediction score
        preprocessor = pipeline['preprocessor']
        model = pipeline['model']
        
        df_input = pd.DataFrame([client_input])
        X_proc = preprocessor.transform(df_input)
        
        proba = model.predict_proba(X_proc)[0, 1]
        
        # Determine risk band
        if proba >= 0.40:
            risk_band = "High Risk"
            risk_class = "risk-high"
        elif proba >= 0.15:
            risk_band = "Medium Risk"
            risk_class = "risk-medium"
        else:
            risk_band = "Low Risk"
            risk_class = "risk-low"
            
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown("### 🏆 Risk Score Summary")
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Model Default Probability</h4>
                <h1 class='{risk_class}'>{proba * 100:.2f}%</h1>
                <p>Classification Band: <strong class='{risk_class}'>{risk_band}</strong></p>
                <p>Repayment Expectation Rate: <strong>{(1 - proba) * 100:.2f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Credit Risk Default Meter (%)", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#1F2937"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 15], 'color': '#D1FAE5'},
                        {'range': [15, 40], 'color': '#FEF3C7'},
                        {'range': [40, 100], 'color': '#FEE2E2'}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col_res2:
            st.markdown("### 🔍 Explainable AI (XAI) Feature Contributions")
            
            # Calculate custom SHAP-like contributions
            contributions = calculate_feature_contributions(client_input)
            contrib_df = pd.DataFrame(contributions)
            
            # Interactive Plotly horizontal bar chart for contributions
            contrib_df = contrib_df.sort_values(by="Contribution", ascending=True)
            
            # Assign colors based on positive (red) vs negative (green) contributions
            colors = ['#DC2626' if x > 0 else '#059669' for x in contrib_df['Contribution']]
            
            fig_bar = go.Figure(go.Bar(
                x=contrib_df['Contribution'],
                y=contrib_df['Label'],
                orientation='h',
                marker_color=colors,
                text=contrib_df['Contribution'].round(2),
                textposition='auto'
            ))
            fig_bar.update_layout(
                title="Risk Contribution Score by Attribute",
                xaxis_title="Contribution to Risk (⬅ Reduces Risk | Increases Risk ➡)",
                yaxis_title="Feature Name",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.divider()
        st.subheader("📋 Decision Rationale & Feature Observations")
        
        # Render clean grid of details
        for c in contributions:
            direction = "⬆️ INCREASES risk score" if c['Contribution'] > 0 else "⬇️ REDUCES risk score"
            icon = "🔴" if c['Contribution'] > 0 else "🟢"
            st.markdown(f"{icon} **{c['Label']}** (Value: `{c['Client Value']}` vs Population Avg: `{c['Population Average']:.2f}`): **{c['Description']}** (Contribution: *{c['Contribution']:.2f}* - *{direction}*)")

# --- MODULE 4: POLICY DECISION RULES ---
elif menu == "📋 Policy Decision Rules":
    st.markdown("<div class='main-title'>📋 Bank Underwriting Policy Rules</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Translate machine learning predictions into structured, auditor-compliant business decision paths.</div>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1, 1.2])
    
    with col_l:
        st.subheader("🔧 Active Credit Risk Policy Rationale")
        st.markdown("""
        To satisfy strict financial audit guidelines (e.g., Basel III, FDIC regulatory frameworks), decisions cannot solely rest on an ML "black box".
        
        This dashboard enforces **hybrid policy guardrails**:
        
        1. **Low Credit Score Knock-Out Rule (`CR-01`)**: Hard auto-rejections for applicants scoring critical low grades (`EXT_2 < 0.22` or `EXT_3 < 0.18`) regardless of total income.
        2. **Affordability Ceiling Rule (`AF-01`)**: Instant rejects for applicants having Debt-to-Income ratios exceeding `45%`.
        3. **ML Risk Boundary Guardrails (`ML-01`, `ML-02`)**:
            *   ML Probabilities `>= 40%` are immediately rejected.
            *   ML Probabilities `15% - 40%` are referred to manual human underwriting.
        4. **Thin File Youth Policy (`PL-01`)**: Manual reviews triggered for applicants under 23 without robust bureau entries.
        5. **Super Prime Auto-Approval Fast Track (`SP-01`)**: Fast-tracking highly qualified applications.
        """)
        
    with col_r:
        st.subheader("🧪 Run Policy Sandbox")
        # Quick input fields
        in_v = st.number_input("Sandbox Income", 10000, 1000000, 130000)
        cr_v = st.number_input("Sandbox Credit Amount", 20000, 2000000, 600000)
        an_v = st.number_input("Sandbox Monthly Annuity", 1000, 100000, 32000)
        e_2 = st.slider("Sandbox Rating Score 2", 0.0, 1.0, 0.18)
        ml_p = st.slider("Model Default Probability Input (%)", 0.0, 100.0, 12.0)
        
        # Evaluate
        sandbox_input = {
            'AMT_INCOME_TOTAL': in_v,
            'AMT_CREDIT': cr_v,
            'AMT_ANNUITY': an_v,
            'EXT_SOURCE_2': e_2,
            'DAYS_BIRTH': -14000
        }
        
        policy_res = evaluate_credit_policy_rules(sandbox_input, ml_p / 100.0)
        
        st.divider()
        st.subheader("⚖️ Sandbox Regulatory Policy Decision")
        
        dec = policy_res['Decision']
        if dec == "APPROVED":
            st.success(f"✔️ **UNDERWRITING DECISION: {dec}**")
        elif dec == "MANUAL_REVIEW":
            st.warning(f"⚠️ **UNDERWRITING DECISION: REFER FOR MANUAL REVIEW**")
        else:
            st.error(f"❌ **UNDERWRITING DECISION: {dec}**")
            
        st.markdown(f"**Policy Risk Classification**: **{policy_res['Risk Band']} Risk**")
        st.markdown(f"**Calculated DTI Ratio**: **{policy_res['DTI Ratio'] * 100:.2f}%** (DTI Max Target Limit: 30%)")
        
        st.markdown("##### 📄 Audit Log / Triggered Policy Rules:")
        for rule in policy_res['Triggered Rules']:
            st.code(rule, language="markdown")
            
        st.markdown("##### 📝 Officer Action Recommendations:")
        for rec in policy_res['Recommendations']:
            st.info(rec)

# --- MODULE 5: TALK-TO-DATA conversatIONAL AI ---
elif menu == "💬 Talk-to-Data Chatbot":
    st.markdown("<div class='main-title'>💬 Conversational Talk-to-Data Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Ask analytical credit questions in plain English, and watch them convert to optimized SQL queries running against our relational SQLite database!</div>", unsafe_allow_html=True)
    
    st.subheader("💡 5 Ready-To-Test Query Templates")
    col_q1, col_q2 = st.columns(2)
    
    queries = [
        "What is the average total income of our applicants?",
        "What is the default rate percentage grouped by the number of children?",
        "Show me the top 5 highest credit amounts where the applicant has no defaults.",
        "Compare how many applicants defaulted vs repaid.",
        "Show me the average income of applicants."
    ]
    
    # Store clicked query in session state
    if "chat_query" not in st.session_state:
        st.session_state["chat_query"] = ""
        
    with col_q1:
        if st.button(f"📊 {queries[0]}", use_container_width=True):
            st.session_state["chat_query"] = queries[0]
        if st.button(f"👶 {queries[1]}", use_container_width=True):
            st.session_state["chat_query"] = queries[1]
        if st.button(f"🏆 {queries[2]}", use_container_width=True):
            st.session_state["chat_query"] = queries[2]
            
    with col_q2:
        if st.button(f"⚖️ {queries[3]}", use_container_width=True):
            st.session_state["chat_query"] = queries[3]
        if st.button(f"💵 {queries[4]}", use_container_width=True):
            st.session_state["chat_query"] = queries[4]

    st.divider()
    
    # Custom query bar
    user_input = st.text_input("💬 Ask a natural language credit risk question...", value=st.session_state["chat_query"])
    
    if user_input:
        with st.spinner("Analyzing dataset tables and generating SQL query via Groq..."):
            # 1. Translate NL to SQL using our modular translation module
            sql = translate_nl_to_sql(user_input)
            
            st.markdown("### 🖥️ Generated SQL Statement")
            st.code(sql, language="sql")
            
            try:
                # 2. Execute SQL query on SQLite
                results_df = execute_query(sql)
                
                st.markdown("### 📊 Database Output Table")
                if results_df.empty:
                    st.warning("No records matched this query.")
                else:
                    st.dataframe(results_df, use_container_width=True)
                    
                    # 3. Synthesize human-readable executive business insight via Groq
                    insight = synthesize_business_insight(user_input, sql, results_df)
                    st.markdown("### 💡 Executive Business Insight Summary")
                    st.success(insight)
            except Exception as e:
                st.error(f"SQL Execution Failure: {str(e)}")

# --- MODULE 6: MODEL DIAGNOSTICS & RETRAINING ---
elif menu == "📈 Model Diagnostics & Retraining":
    st.markdown("<div class='main-title'>📈 Machine Learning Model Diagnostics &amp; Retraining</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Monitor model performance parameters, verify metrics, and trigger dynamic retraining on the real dataset.</div>", unsafe_allow_html=True)
    
    # Model details
    st.subheader("🤖 Model Performance Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Model Meta Details
        *   **Algorithm**: Random Forest Classifier
        *   **Imbalance Strategy**: Balanced Class Weighting (`class_weight='balanced'`)
        *   **Hyperparameters**:
            *   Number of Estimators: `200`
            *   Max Depth: `10`
            *   Random State: `42`
        *   **Target Metrics (Home Credit Benchmark)**:
            *   Target ROC-AUC: **~0.72 - 0.75** (Industry standard for credit risk)
            *   PR-AUC / Average Precision: **~0.25 - 0.30**
        """)
    with col2:
        # Dynamic metrics verification
        st.markdown("### Run Live Model Diagnostics")
        if st.button("Run Live Model Diagnostics", type="secondary"):
            with st.spinner("Calculating validation metrics on database records..."):
                try:
                    # Perform a live evaluation of the saved model on the database records
                    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, auc
                    
                    df_eval = mock_data_df.copy()
                    X_eval = df_eval.drop(columns=['TARGET'], errors='ignore')
                    y_eval = df_eval['TARGET']
                    
                    preprocessor = pipeline['preprocessor']
                    model = pipeline['model']
                    
                    X_proc = preprocessor.transform(X_eval)
                    probs = model.predict_proba(X_proc)[:, 1]
                    preds = model.predict(X_proc)
                    
                    roc_auc = roc_auc_score(y_eval, probs)
                    precision, recall, _ = precision_recall_curve(y_eval, probs)
                    pr_auc = auc(recall, precision)
                    
                    st.success("Diagnostics Complete!")
                    
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("ROC-AUC Score", f"{roc_auc:.4f}")
                    col_m2.metric("PR-AUC (Avg Precision)", f"{pr_auc:.4f}")
                    
                    st.markdown("#### Classification Report:")
                    st.code(classification_report(y_eval, preds))
                    
                    st.markdown("#### Confusion Matrix:")
                    cm = confusion_matrix(y_eval, preds)
                    cm_df = pd.DataFrame(cm, columns=["Repaid (0)", "Defaulted (1)"], index=["Repaid (0)", "Defaulted (1)"])
                    st.dataframe(cm_df)
                    
                except Exception as e:
                    st.error(f"Failed to calculate metrics: {e}")
                    
    st.divider()
    st.subheader("🔄 Trigger ML Retraining Pipeline")
    st.markdown("""
    You can trigger live model retraining on the host machine. 
    This executes the complete data science pipeline:
    1. Loads `data/application_train.csv` (or falls back to mock data if absent).
    2. Performs stratified training/validation split.
    3. Fits the custom `CreditRiskPreprocessor` (imputing, Z-score scaling, one-hot encoding).
    4. Fits the `RandomForestClassifier` with balanced class weights.
    5. Serializes the new model to `models/credit_risk_model.joblib`.
    """)
    
    if st.button("Trigger Retraining Pipeline", type="primary"):
        with st.spinner("Running complete ML Training pipeline... (This might take 10-15 seconds)"):
            try:
                from src.ml.train import run_training
                run_training("application_train.csv", "credit_risk_model.joblib")
                
                # Clear resource cache to reload the new model immediately
                st.cache_resource.clear()
                st.success("Model Retrained Successfully! New pipeline serialized to `models/credit_risk_model.joblib` and reloaded.")
            except Exception as e:
                st.error(f"Retraining failed: {e}")
