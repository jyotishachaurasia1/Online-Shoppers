import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Shopper Intent Classifier",
                   page_icon="🛍️", layout="wide")

st.markdown("""
<style>
.hero {
    background: linear-gradient(90deg, #0f766e 0%, #14b8a6 100%);
    padding: 1.4rem 1.8rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: 0.3rem 0 0 0; font-size: 1rem; opacity: 0.95;}
div[data-testid="stMetric"] {
    background-color: #f0fdfa;
    border: 1px solid #99f6e4;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
}
</style>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl",
}
TARGET = "Revenue"
CLASS_NAMES = ["No Purchase", "Purchase"]

METRIC_HELP = {
    "Accuracy": "Share of all sessions predicted correctly.",
    "AUC": "How well the model ranks buyers above non-buyers (0.5 = random, 1 = perfect).",
    "Precision": "Of sessions predicted as 'Purchase', how many really bought.",
    "Recall": "Of all real buyers, how many the model caught.",
    "F1": "Balance between Precision and Recall.",
    "MCC": "Balanced score from -1 to 1; fair even when classes are imbalanced.",
}


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)


def get_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # AUC needs both classes in the uploaded file
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def run_all_models(X, y):
    preds, results = {}, {}
    for name, path in MODEL_FILES.items():
        model = load_model(path)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        preds[name] = y_pred
        results[name] = get_metrics(y, y_pred, y_prob)
    return preds, pd.DataFrame(results).T


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🛍️ Will This Visitor Buy?</h1>
  <p>Shopper intent classification on the UCI Online Shoppers dataset,
  comparing 5 machine learning models on unseen test data.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("📂 Data & Model")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()),
                                  index=1)

try:
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button("⬇️ Download sample test data", f,
                                   file_name="test_data.csv", mime="text/csv")
except FileNotFoundError:
    pass

st.sidebar.markdown("---")
st.sidebar.markdown("**ℹ️ About**")
st.sidebar.caption(
    "Dataset: UCI Online Shoppers Purchasing Intention "
    "(12,205 sessions, 17 features, ~15.6% buyers).\n\n"
    "Models are pre-trained and only loaded here; "
    "they are not retrained on the uploaded file.\n\n"
    "Made by Jyotisha Chaurasia – BITS Pilani M.Tech (AIML)."
)

# ------------------------------------------------------------------
# Waiting for upload
# ------------------------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload a test CSV from the sidebar to see the results. "
            "No file? Use **Download sample test data** first.")
    st.markdown("""
#### How to use
1. Download the sample test data (or use your own file with the same columns).
2. Upload it in the sidebar.
3. Pick a model from the dropdown to see its metrics, confusion matrix and report.
4. Open the **Model Leaderboard** tab to compare all 5 models.
""")
    st.stop()

# ------------------------------------------------------------------
# Read and validate the file
# ------------------------------------------------------------------
data = pd.read_csv(uploaded_file)

if TARGET not in data.columns:
    st.error(f"The file must contain the target column '{TARGET}'.")
    st.stop()

for col in ["Weekend", TARGET]:
    if col in data.columns and data[col].dtype == bool:
        data[col] = data[col].astype(int)

X = data.drop(columns=[TARGET])
y = data[TARGET]

try:
    all_preds, leaderboard = run_all_models(X, y)
except Exception as err:
    st.error("The models could not read this file. Make sure it has the same "
             f"columns as the sample test data.\n\nDetails: {err}")
    st.stop()

# Dataset summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sessions", f"{len(data):,}")
c2.metric("Features", X.shape[1])
c3.metric("Buyers", f"{int(y.sum()):,}")
c4.metric("Buyer share", f"{y.mean() * 100:.1f}%")

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_metrics, tab_matrix, tab_board, tab_data = st.tabs(
    ["📊 Model Metrics", "🧮 Confusion Matrix & Report",
     "🏆 Model Leaderboard", "🔍 Data Preview"]
)

selected = leaderboard.loc[model_name]
y_pred = all_preds[model_name]

with tab_metrics:
    st.subheader(f"Evaluation metrics – {model_name}")
    cols = st.columns(6)
    for col, metric in zip(cols, METRIC_HELP):
        col.metric(metric, f"{selected[metric]:.4f}", help=METRIC_HELP[metric])
    with st.expander("What do these metrics mean?"):
        for metric, text in METRIC_HELP.items():
            st.markdown(f"- **{metric}:** {text}")
        st.caption("Only ~15% of sessions are purchases, so MCC and F1 "
                   "are more reliable than accuracy here.")

with tab_matrix:
    st.subheader(f"How {model_name} classified each session")
    left, right = st.columns([1, 1.2])
    with left:
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"Caught {tp} of {tp + fn} buyers · "
                   f"{fp} false alarms · {fn} buyers missed")
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=[0, 1],
                                       target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4))

with tab_board:
    st.subheader("All 5 models on the uploaded data")
    board = leaderboard.round(4).sort_values("MCC", ascending=False)
    st.dataframe(board.style.highlight_max(axis=0, color="#99f6e4")
                 .format("{:.4f}"))

    best = board.index[0]
    st.success(f"🏆 Best model on this data (by MCC): **{best}** "
               f"(MCC {board.loc[best, 'MCC']:.4f}, F1 {board.loc[best, 'F1']:.4f})")

    chart_metric = st.radio("Compare models by", list(METRIC_HELP),
                            index=5, horizontal=True)
    st.bar_chart(leaderboard[chart_metric].sort_values(ascending=False),
                 color="#0f766e")

with tab_data:
    st.subheader("Uploaded data")
    st.dataframe(data.head(20))
    st.markdown("**Purchase vs No Purchase in this file**")
    st.bar_chart(y.map({0: "No Purchase", 1: "Purchase"}).value_counts(),
                 color="#14b8a6")
