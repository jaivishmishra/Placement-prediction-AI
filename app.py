import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Placement Prediction AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS (Sleek, Professional Glassmorphism Theme)
# =========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top left, #0B0F15, #0A0E13);
        color: #E0E0E0;
    }
    

    /* Adjust top padding so your tabs sit snugly */
    .block-container {
        padding-top: 3.5rem !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        color: #FFFFFF;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    .glass-card {
        background: rgba(19, 24, 32, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.5);
    }

    .metric-card {
        background: #131820;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 24px;
        padding: 25px 20px;
        text-align: center;
        color: #FFFFFF;
    }

    .metric-card h1 {
        font-size: 38px;
        font-weight: 700;
        margin: 5px 0;
        color: #2ECC71;
    }

    .metric-card p {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #A0AEC0;
        margin-bottom: 10px;
    }

    .insight-card {
        background: #131820;
        padding: 20px 24px;
        border-left: 4px solid #2ECC71;
        border-radius: 16px;
        margin-bottom: 12px;
        font-size: 15px;
        color: #CBD5E0;
        transition: background 0.2s;
    }

    .insight-card:hover {
        background: #1A212C;
    }

    .stButton > button {
        width: 100%;
        border-radius: 16px;
        height: 3.2em;
        background: linear-gradient(135deg, #2ECC71, #27AE60);
        color: white;
        font-size: 18px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #27AE60, #219A52);
        box-shadow: 0 6px 20px rgba(46, 204, 113, 0.3);
        transform: scale(1.02);
    }

    section[data-testid="stSidebar"] {
        background-color: #0B0F15;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    .css-1d391kg, .css-1lcbmhc {
        background-color: transparent;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #131820;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD ASSETS (Scaler, Models) with Caching
# =========================================
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_model(model_name):
    if model_name == "Random Forest":
        return joblib.load("rf.pkl")
    elif model_name == "Logistic Regression":
        return joblib.load("logistic.pkl")
    else:
        return joblib.load("svm.pkl")

scaler = load_scaler()

# =========================================
# SIDEBAR (Mission Control)
# =========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=80)
    st.title("Placement AI")

    st.markdown("""
    <div style="color:#A0AEC0; font-size:14px; margin-bottom:20px;">
    AI-powered placement prediction & skill analytics
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "🤖 Choose ML Model",
        ["Random Forest", "Logistic Regression", "SVM"],
        help="Different models offer different trade-offs between interpretability and accuracy."
    )

    model = load_model(model_choice)

    st.markdown("---")

    # Model performance metrics (static, could be from a file)
    with st.expander("📊 Model Performance"):
        metrics = {
            "Random Forest": {"Accuracy": "94%", "Precision": "93%", "Recall": "92%"},
            "Logistic Regression": {"Accuracy": "89%", "Precision": "88%", "Recall": "87%"},
            "SVM": {"Accuracy": "91%", "Precision": "90%", "Recall": "89%"}
        }
        m = metrics[model_choice]
        for k, v in m.items():
            st.metric(label=k, value=v)

    st.markdown("---")
    st.caption("© 2026 Placement AI • Built with Streamlit & Scikit-Learn")

# =========================================
# TABS: Structure for Professional UX
# =========================================
tab1, tab2, tab3 = st.tabs(["🏠 Single Prediction", "📂 Bulk CSV Analysis", "📊 Model Insights"])

# =========================================
# TAB 1: SINGLE STUDENT PREDICTION
# =========================================
with tab1:
    st.markdown('<div class="glass-card"><h2>🎓 Student Profile Input</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        iq = st.slider("🧠 IQ Score", 50, 160, 100, help="Cognitive ability index")
        cgpa = st.slider("📚 CGPA", 4.0, 10.0, 7.0, step=0.1)
    with col2:
        communication = st.slider("🗣️ Communication Skills", 1, 10, 5, help="1 = low, 10 = excellent")
        projects = st.slider("🛠️ Projects Completed", 0, 10, 2)

    # Summary with metric cards
    st.markdown("## 🧾 Student Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><p>IQ</p><h1>{iq}</h1></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><p>CGPA</p><h1>{cgpa:.1f}</h1></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><p>Communication</p><h1>{communication}</h1></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><p>Projects</p><h1>{projects}</h1></div>', unsafe_allow_html=True)

    # Profile Strength Gauge (replaces progress bar)
    profile_score = (cgpa * 10 + communication * 5 + projects * 10 + iq / 2) / 4
    profile_normalized = min(max(profile_score / 82.5 * 100, 0), 100)  # Scale to 0-100

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=profile_normalized,
        title={'text': "Profile Strength Index", 'font': {'color': "white"}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': "#2ECC71"},
            'bgcolor': "#131820",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.1)"
        },
        number={'suffix': "%", 'font': {'color': "white"}}
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Prediction Button with loading state
    predict_btn = st.button("🚀 Predict Placement", type="primary")

    if predict_btn:
        with st.spinner("AI analyzing profile..."):
            input_data = np.array([[iq, cgpa, communication, projects]])
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            placed_prob = probability[0][1] * 100
            not_placed_prob = probability[0][0] * 100

            # Store in session state for persistence
            st.session_state['prediction'] = prediction[0]
            st.session_state['placed_prob'] = placed_prob
            st.session_state['not_placed_prob'] = not_placed_prob
            st.session_state['input_scaled'] = input_scaled
            st.session_state['input_values'] = (iq, cgpa, communication, projects)

    # If we have a prediction (either just made or from session state), show results
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        placed_prob = st.session_state['placed_prob']
        not_placed_prob = st.session_state['not_placed_prob']
        iq, cgpa, communication, projects = st.session_state['input_values']

        st.markdown("---")
        st.markdown("## 📌 Prediction Result")

        if prediction == 1:
            st.success("✅ Student is **LIKELY TO BE PLACED**")
        else:
            st.error("❌ Student is **UNLIKELY TO BE PLACED**")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f'''
            <div class="metric-card">
                <p>📈 Placement Probability</p>
                <h1>{placed_prob:.1f}%</h1>
            </div>
            ''', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'''
            <div class="metric-card">
                <p>❌ Rejection Probability</p>
                <h1>{not_placed_prob:.1f}%</h1>
            </div>
            ''', unsafe_allow_html=True)

        # Gauge for placement probability
        fig_prob = go.Figure(go.Indicator(
            mode="gauge+number",
            value=placed_prob,
            title={'text': "Placement Likelihood", 'font': {'color': "white"}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': "#2ECC71" if placed_prob > 70 else "#F39C12" if placed_prob > 40 else "#E74C3C"},
                'bgcolor': "#131820",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(231, 76, 60, 0.3)'},
                    {'range': [40, 70], 'color': 'rgba(243, 156, 18, 0.3)'},
                    {'range': [70, 100], 'color': 'rgba(46, 204, 113, 0.3)'}
                ]
            },
            number={'suffix': "%", 'font': {'color': "white"}}
        ))
        fig_prob.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_prob, use_container_width=True)

        # Risk Analysis
        if placed_prob < 40:
            risk_level = "High Risk"
            risk_color = "#E74C3C"
        elif placed_prob < 70:
            risk_level = "Moderate Potential"
            risk_color = "#F39C12"
        else:
            risk_level = "Strong Potential"
            risk_color = "#2ECC71"

        st.markdown(f'''
        <div class="glass-card" style="border-left: 4px solid {risk_color};">
            <h4>⚠️ Risk Analysis: {risk_level}</h4>
        </div>
        ''', unsafe_allow_html=True)

        # Actionable Recommendations
        st.markdown("### 📋 Personalized Recommendations")
        recs = []
        if cgpa < 7:
            recs.append("📘 Improve CGPA – focus on core subjects and seek tutoring.")
        if communication < 5:
            recs.append("🗣️ Enhance communication – join Toastmasters, practice mock interviews.")
        if projects < 2:
            recs.append("🛠️ Build more real-world projects – contribute to GitHub, participate in hackathons.")
        if iq < 100:
            recs.append("🧠 Boost aptitude – practice logical reasoning and puzzles regularly.")
        if not recs:
            recs.append("✅ Your profile is well-rounded. Keep up the great work!")

        for r in recs:
            st.markdown(f'<div class="insight-card">{r}</div>', unsafe_allow_html=True)

        # AI Career Analysis
        with st.expander("🤖 AI Career Analysis", expanded=False):
            analysis = []
            strength_score = 0
            if iq >= 120:
                analysis.append("Excellent analytical ability detected.")
                strength_score += 25
            elif iq >= 100:
                analysis.append("Good problem-solving capability.")
                strength_score += 15
            else:
                analysis.append("Aptitude improvement recommended.")
                strength_score += 5

            if cgpa >= 8.5:
                analysis.append("Strong academic consistency.")
                strength_score += 25
            elif cgpa >= 7:
                analysis.append("Academics are above average.")
                strength_score += 15
            else:
                analysis.append("Academic performance may hurt placements.")
                strength_score += 5

            if communication >= 8:
                analysis.append("Excellent communication skills.")
                strength_score += 25
            elif communication >= 5:
                analysis.append("Communication skills are acceptable.")
                strength_score += 15
            else:
                analysis.append("Communication skills need improvement.")
                strength_score += 5

            if projects >= 5:
                analysis.append("Strong practical project portfolio.")
                strength_score += 25
            elif projects >= 2:
                analysis.append("Moderate project experience.")
                strength_score += 15
            else:
                analysis.append("More real-world projects recommended.")
                strength_score += 5

            if strength_score >= 80:
                verdict = "Highly Employable Profile"
            elif strength_score >= 60:
                verdict = "Moderately Competitive Profile"
            else:
                verdict = "High Risk Placement Profile"

            st.markdown(f'''
            <div class="glass-card">
                <h3>🧠 AI Final Verdict</h3>
                <h1 style="color:#2ECC71;">{verdict}</h1>
                <p>Profile Intelligence Score: {strength_score}/100</p>
            </div>
            ''', unsafe_allow_html=True)
            for point in analysis:
                st.markdown(f'<div class="insight-card">✅ {point}</div>', unsafe_allow_html=True)

        # Explainability (SHAP for Random Forest)
        if model_choice == "Random Forest":

            # Inside the expander:
            with st.expander("🧠 Why This Prediction? (SHAP Explainability)", expanded=False):
                st.markdown("### Feature contribution to prediction")
                try:
                    # Use the modern explainer interface
                    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
                    shap_values = explainer(st.session_state['input_scaled'])
                    
                    # Get SHAP values for the positive class (index 1)
                    sv_pos = shap_values[..., 1]           # shape (1, n_features)
                    expected_value = explainer.expected_value[1]  # base value for positive class

                    feature_names = ["IQ", "CGPA", "Communication", "Projects"]
                    explanation = shap.Explanation(
                        values=sv_pos[0],
                        base_values=expected_value,
                        data=st.session_state['input_scaled'][0],
                        feature_names=feature_names
                    )
                    
                    # Create the waterfall plot, don't show it, then grab the figure
                    shap.plots.waterfall(explanation, show=False)
                    fig_shap = plt.gcf()   # get current figure
                    st.pyplot(fig_shap)
                    plt.clf()              # clear the figure to avoid memory leaks
                    
                except Exception as e:
                    st.warning(f"SHAP visualization currently unavailable: {e}")
        else:
            st.markdown('''
            <div class="glass-card">
                <p>⚠️ Explainable AI (SHAP) is only available for Random Forest model.</p>
            </div>
            ''', unsafe_allow_html=True)

# =========================================
# TAB 2: BULK CSV ANALYSIS
# =========================================
with tab2:
    st.markdown('<div class="glass-card"><h2>📂 Bulk Student Prediction</h2><p>Upload a CSV with IQ, CGPA, Communication, Projects columns.</p></div>', unsafe_allow_html=True)

    # Template download
    template_df = pd.DataFrame(columns=["IQ", "CGPA", "Communication_Skills", "Projects_Completed"])
    st.download_button(
        "⬇️ Download CSV Template",
        data=template_df.to_csv(index=False),
        file_name="student_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Student CSV", type=["csv"])
    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)
        # Standardize column names
        bulk_df.columns = bulk_df.columns.str.strip().str.lower().str.replace(' ', '_')
        col_map = {
            'iq': 'IQ',
            'cgpa': 'CGPA',
            'communication': 'Communication_Skills',
            'communication_skills': 'Communication_Skills',
            'projects': 'Projects_Completed',
            'projects_completed': 'Projects_Completed'
        }
        bulk_df.rename(columns=col_map, inplace=True)

        required = ['IQ', 'CGPA', 'Communication_Skills', 'Projects_Completed']
        missing = [c for c in required if c not in bulk_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}. Please match the template.")
        else:
            st.success("CSV loaded successfully!")
            st.dataframe(bulk_df.head(), use_container_width=True)

            # Predict
            with st.spinner("Running predictions on batch..."):
                input_bulk = bulk_df[required]
                input_scaled = scaler.transform(input_bulk)
                preds = model.predict(input_scaled)
                probs = model.predict_proba(input_scaled)[:, 1] * 100

                bulk_df['Prediction'] = preds
                bulk_df['Placement_Probability'] = probs
                bulk_df['Prediction_Label'] = bulk_df['Prediction'].map({0: 'Not Placed', 1: 'Placed'})

            st.markdown("### 📊 Prediction Results")
            st.dataframe(bulk_df, use_container_width=True)

            csv = bulk_df.to_csv(index=False)
            st.download_button("⬇️ Download Predictions CSV", data=csv, file_name="placement_predictions.csv", mime="text/csv")

            # Visual analytics
            st.markdown("### 📈 Student Vector Analytics")
            fig1 = px.scatter(bulk_df, x="CGPA", y="IQ", color="Prediction_Label", size="Placement_Probability",
                              hover_data=["Communication_Skills", "Projects_Completed"], title="CGPA vs IQ")
            fig1.update_layout(paper_bgcolor="#131820", plot_bgcolor="#131820", font_color="white")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.scatter(bulk_df, x="Communication_Skills", y="Projects_Completed", color="Prediction_Label",
                              size="Placement_Probability", hover_data=["IQ", "CGPA"], title="Communication vs Projects")
            fig2.update_layout(paper_bgcolor="#131820", plot_bgcolor="#131820", font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

            # Distribution pie
            placement_counts = bulk_df['Prediction_Label'].value_counts()
            fig3 = px.pie(names=placement_counts.index, values=placement_counts.values, title="Placement Distribution", hole=0.5)
            fig3.update_layout(paper_bgcolor="#131820", font_color="white")
            st.plotly_chart(fig3, use_container_width=True)

            # Histogram
            fig4 = px.histogram(bulk_df, x="Placement_Probability", color="Prediction_Label", nbins=20,
                                title="Probability Distribution")
            fig4.update_layout(paper_bgcolor="#131820", plot_bgcolor="#131820", font_color="white")
            st.plotly_chart(fig4, use_container_width=True)

            # Correlation heatmap
            st.markdown("### 🔥 Correlation Heatmap")
            corr = bulk_df[required].corr()
            fig5 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu")
            fig5.update_layout(paper_bgcolor="#131820", font_color="white")
            st.plotly_chart(fig5, use_container_width=True)

# =========================================
# TAB 3: MODEL INSIGHTS (Feature Importance)
# =========================================
with tab3:
    st.markdown('<div class="glass-card"><h2>📊 Model Insights</h2></div>', unsafe_allow_html=True)
    if model_choice == "Random Forest":
        importance = model.feature_importances_
        feature_names = ["IQ", "CGPA", "Communication", "Projects"]
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", text="Importance", color="Importance",
                     color_continuous_scale="Greens")
        fig.update_layout(paper_bgcolor="#131820", plot_bgcolor="#131820", font_color="white", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="glass-card"><p>Feature importance is only available for Random Forest.</p></div>', unsafe_allow_html=True)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("Built with Streamlit • Scikit-Learn • Plotly • SHAP • AI Analytics")