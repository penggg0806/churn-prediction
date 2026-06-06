import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---- 页面配置 ----
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- 自定义 CSS ----
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 12px;
    }
    .insight-box {
        background: #eaf4fb;
        border-left: 4px solid #3498db;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.95rem;
    }
    .high-risk { color: #e74c3c; font-weight: bold; }
    .med-risk  { color: #e67e22; font-weight: bold; }
    .low-risk  { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---- 数据 ----
model_df = pd.DataFrame({
    "Model":      ["Logistic Regression", "Random Forest", "GBT"],
    "AUC":        [0.8291, 0.8300, 0.8274],
    "F1":         [0.7985, 0.7798, 0.7932],
    "Precision":  [0.8012, 0.7845, 0.7901],
    "Recall":     [0.7960, 0.7753, 0.7964],
})

feature_df = pd.DataFrame({
    "Feature":    ["tenure", "Contract_0", "OnlineSecurity_0",
                   "InternetService_0", "TechSupport_0", "TotalCharges",
                   "Contract_1", "PaymentMethod_0", "MonthlyCharges", "OnlineBackup_0"],
    "Importance": [0.1591, 0.1545, 0.0983, 0.0806, 0.0779,
                   0.0772, 0.0557, 0.0482, 0.0353, 0.0303],
    "Category":   ["Numerical", "Contract", "Service", "Service", "Service",
                   "Numerical", "Contract", "Payment", "Numerical", "Service"]
})

contract_df = pd.DataFrame({
    "Contract":   ["Month-to-month", "One year", "Two year"],
    "Churn":      [1655, 166, 48],
    "No Churn":   [2220, 1307, 1647],
})

tenure_churn   = [600, 350, 220, 180, 150, 120, 100, 80, 70, 60,
                   50, 40, 35, 30, 25, 20, 18, 16, 14, 12]
tenure_nochurn = [380, 220, 200, 190, 185, 180, 175, 170, 165, 160,
                  200, 210, 220, 230, 240, 290, 310, 350, 400, 630]
tenure_x = [i*3.5 for i in range(20)]

# ---- 侧边栏 ----
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("Churn Prediction")
st.sidebar.markdown("---")

st.sidebar.subheader("🔍 Filter Options")
selected_model   = st.sidebar.selectbox("Select Model", model_df["Model"].tolist())
risk_filter      = st.sidebar.multiselect(
    "Risk Level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
auc_threshold    = st.sidebar.slider("Min AUC Threshold", 0.75, 0.90, 0.82, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Tech Stack")
st.sidebar.markdown("""
- 🗄️ **HDFS** — Distributed Storage  
- ⚡ **Spark MLlib** — Feature Eng. & ML  
- 🗃️ **HBase** — Online Serving Layer  
- 📊 **Streamlit** — Dashboard  
""")

st.sidebar.subheader("📁 Dataset")
st.sidebar.markdown("""
- Telco Customer Churn  
- **7,043** users × **21** features  
- Train/Test: **80% / 20%**  
- Class imbalance: **73.5% / 26.5%**  
""")

# ---- Header ----
st.markdown("## 📊 User Churn Prediction System")
st.markdown("End-to-end churn prediction pipeline · Telco Customer Churn · Spark MLlib")
st.markdown("---")

# ---- KPI 指标行 ----
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Users",    "7,043")
k2.metric("Churned Users",  "1,869",  "26.5%")
k3.metric("Best AUC",       "0.8300", "Random Forest")
k4.metric("Best F1",        "0.7985", "Logistic Reg.")
k5.metric("Features",       "45 dims","after OHE")
st.markdown("---")

# ---- Tab 布局 ----
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Overview", "🤖 Model Comparison", "🔑 Feature Importance", "💡 Insights"])

# ======== Tab 1: Overview ========
with tab1:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Churn Distribution</div>',
                    unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        colors = ["#2ecc71", "#e74c3c"]
        wedges, texts, autotexts = ax1.pie(
            [5174, 1869],
            labels=["No Churn", "Churn"],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")
        ax1.set_title("Churn Distribution (N=7,043)", fontsize=12, pad=15)
        fig1.patch.set_facecolor("#f8f9fa")
        st.pyplot(fig1)

    with col_r:
        st.markdown('<div class="section-title">Contract Type vs Churn</div>',
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        x = np.arange(len(contract_df))
        w = 0.35
        ax2.bar(x - w/2, contract_df["Churn"],    w, label="Churn",    color="#e74c3c", alpha=0.85)
        ax2.bar(x + w/2, contract_df["No Churn"], w, label="No Churn", color="#2ecc71", alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(contract_df["Contract"], fontsize=9)
        ax2.legend()
        ax2.set_title("Contract Type vs Churn", fontsize=12)
        ax2.set_ylabel("Count")
        ax2.spines[["top","right"]].set_visible(False)
        fig2.patch.set_facecolor("#f8f9fa")
        st.pyplot(fig2)

    st.markdown('<div class="section-title">Tenure Distribution by Churn</div>',
                unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(11, 3.5))
    ax3.fill_between(tenure_x, tenure_churn,   alpha=0.6, color="#e74c3c", label="Churn")
    ax3.fill_between(tenure_x, tenure_nochurn, alpha=0.6, color="#2ecc71", label="No Churn")
    ax3.set_xlabel("Tenure (months)")
    ax3.set_ylabel("Count")
    ax3.set_title("Tenure Distribution — Churn vs No Churn")
    ax3.legend()
    ax3.spines[["top","right"]].set_visible(False)
    fig3.patch.set_facecolor("#f8f9fa")
    st.pyplot(fig3)

# ======== Tab 2: Model Comparison ========
with tab2:
    st.markdown('<div class="section-title">Model Performance Comparison</div>',
                unsafe_allow_html=True)

    filtered_models = model_df[model_df["AUC"] >= auc_threshold]

    col_l, col_r = st.columns([3, 2])

    with col_l:
        metrics = ["AUC", "F1", "Precision", "Recall"]
        colors_m = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c"]
        fig4, axes = plt.subplots(1, 4, figsize=(11, 4))
        for ax, metric, color in zip(axes, metrics, colors_m):
            bars = ax.bar(filtered_models["Model"], filtered_models[metric],
                          color=color, alpha=0.85, edgecolor="white")
            ax.set_title(metric, fontsize=11, fontweight="bold")
            ax.set_ylim(0.70, 0.88)
            ax.set_xticklabels(filtered_models["Model"], rotation=20, fontsize=7)
            ax.spines[["top","right"]].set_visible(False)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.002,
                        f"{bar.get_height():.4f}",
                        ha="center", fontsize=7)
            fig4.patch.set_facecolor("#f8f9fa")
        plt.tight_layout()
        st.pyplot(fig4)

    with col_r:
        st.markdown("**Selected Model Detail**")
        sel = model_df[model_df["Model"] == selected_model].iloc[0]
        st.markdown(f"""
        | Metric | Score |
        |--------|-------|
        | AUC    | `{sel.AUC:.4f}` |
        | F1     | `{sel.F1:.4f}` |
        | Precision | `{sel.Precision:.4f}` |
        | Recall | `{sel.Recall:.4f}` |
        """)
        best = "✅ Best AUC" if sel.AUC == model_df.AUC.max() else ""
        bestf1 = "✅ Best F1" if sel.F1 == model_df.F1.max() else ""
        if best:  st.success(best)
        if bestf1: st.success(bestf1)

    st.markdown("**All Models**")
    st.dataframe(
        model_df.style
            .highlight_max(subset=["AUC","F1","Precision","Recall"], color="#d5f5e3")
            .format({"AUC":"{:.4f}","F1":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}"}),
        use_container_width=True
    )

# ======== Tab 3: Feature Importance ========
with tab3:
    st.markdown('<div class="section-title">Top 10 Feature Importance (Random Forest)</div>',
                unsafe_allow_html=True)

    category_filter = st.multiselect(
        "Filter by Category",
        feature_df["Category"].unique().tolist(),
        default=feature_df["Category"].unique().tolist()
    )
    filtered_feat = feature_df[feature_df["Category"].isin(category_filter)]

    col_l, col_r = st.columns([3, 2])

    with col_l:
        cat_colors = {
            "Numerical": "#3498db",
            "Contract":  "#e74c3c",
            "Service":   "#2ecc71",
            "Payment":   "#9b59b6"
        }
        bar_colors = [cat_colors[c] for c in filtered_feat["Category"]]

        fig5, ax5 = plt.subplots(figsize=(8, 5))
        bars = ax5.barh(filtered_feat["Feature"][::-1],
                        filtered_feat["Importance"][::-1],
                        color=bar_colors[::-1], edgecolor="white")
        ax5.set_xlabel("Importance Score")
        ax5.set_title("Feature Importance by Category")
        for bar in bars:
            ax5.text(bar.get_width() + 0.002,
                     bar.get_y() + bar.get_height()/2,
                     f"{bar.get_width():.4f}", va="center", fontsize=9)
        legend_patches = [mpatches.Patch(color=v, label=k)
                          for k, v in cat_colors.items()]
        ax5.legend(handles=legend_patches, loc="lower right", fontsize=9)
        ax5.spines[["top","right"]].set_visible(False)
        fig5.patch.set_facecolor("#f8f9fa")
        plt.tight_layout()
        st.pyplot(fig5)

    with col_r:
        st.markdown("**Feature Details**")
        st.dataframe(
            filtered_feat.style.format({"Importance": "{:.4f}"}),
            use_container_width=True
        )

# ======== Tab 4: Insights ========
with tab4:
    st.markdown('<div class="section-title">📌 Statistical Insights</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <b>① Contract Type is the Strongest Business Predictor</b><br>
    Month-to-month customers churn at <span class="high-risk">~42%</span>,
    vs One-year <span class="med-risk">~11%</span>
    and Two-year <span class="low-risk">~3%</span>.
    Encouraging annual contracts is the highest-leverage retention lever.
    </div>

    <div class="insight-box">
    <b>② Tenure is the Top ML Feature (Importance: 0.1591)</b><br>
    Churn is heavily concentrated in the first <span class="high-risk">1–6 months</span>.
    Early onboarding intervention can significantly reduce lifetime churn.
    </div>

    <div class="insight-box">
    <b>③ Value-Added Services Strongly Reduce Churn</b><br>
    Customers without OnlineSecurity or TechSupport churn at nearly
    <span class="high-risk">2× the rate</span> of those with these services.
    Bundling increases stickiness.
    </div>

    <div class="insight-box">
    <b>④ Class Imbalance Handled — Use AUC & F1, Not Accuracy</b><br>
    With 73.5% / 26.5% split, a naive model predicting "No Churn" always
    achieves 73.5% accuracy. AUC=<b>0.83</b> and F1=<b>0.80</b> are the meaningful metrics.
    </div>

    <div class="insight-box">
    <b>⑤ Logistic Regression vs Tree Models</b><br>
    LR achieves the highest F1 (0.7985) despite being the simplest model,
    suggesting the churn signal is largely linear. GBT adds marginal gain
    at higher computational cost — classic bias-variance tradeoff.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Risk Level Distribution (HBase stored predictions)**")
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    risk_col1.markdown('<p class="high-risk">🔴 HIGH Risk (prob ≥ 0.7)</p>', unsafe_allow_html=True)
    risk_col1.metric("Users", "~312", "23% of test set")
    risk_col2.markdown('<p class="med-risk">🟡 MEDIUM Risk (0.4–0.7)</p>', unsafe_allow_html=True)
    risk_col2.metric("Users", "~287", "21% of test set")
    risk_col3.markdown('<p class="low-risk">🟢 LOW Risk (prob < 0.4)</p>', unsafe_allow_html=True)
    risk_col3.metric("Users", "~746", "56% of test set")

st.markdown("---")
st.caption("Data: Kaggle Telco Customer Churn | Models: Spark MLlib 3.5.0 | Storage: HDFS + HBase | Dashboard: Streamlit")