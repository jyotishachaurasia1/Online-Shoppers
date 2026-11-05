import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Online Shopper Purchase Predictor",
                   page_icon="🛒", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl",
}
TARGET = "Revenue"


@st.cache_resource
def load_model(path):
    return joblib.load(path)

def get_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("🛒 Online Shoppers Purchase Prediction")
st.write("Predict whether an online shopping session ends in a purchase, "
         "using 5 classification models trained on the UCI Online Shoppers dataset.")

st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

if uploaded_file is None:
    st.info("👈 Upload test_data.csv from the sidebar to begin.")
    st.stop()

data = pd.read_csv(uploaded_file)

if TARGET not in data.columns:
    st.error(f"The file must contain the target column '{TARGET}'.")
    st.stop()

for col in ["Weekend", TARGET]:
    if col in data.columns and data[col].dtype == bool:
        data[col] = data[col].astype(int)

X = data.drop(columns=[TARGET])
y = data[TARGET]

st.subheader("📄 Uploaded Data Preview")
st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
st.dataframe(data.head())

model = load_model(MODEL_FILES[model_name])
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]
metrics = get_metrics(y, y_pred, y_prob)

st.subheader(f"📊 Evaluation Metrics: {model_name}")
metric_cols = st.columns(6)
for col, (name, value) in zip(metric_cols, metrics.items()):
    col.metric(name, f"{value:.4f}")


    left, right = st.columns(2)

with left:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["No Purchase", "Purchase"],
                yticklabels=["No Purchase", "Purchase"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification Report**")
    report = classification_report(y, y_pred, output_dict=True,
                                   target_names=["No Purchase", "Purchase"])
    st.dataframe(pd.DataFrame(report).T.round(4))

    st.subheader("🏆 Comparison of All Models on Uploaded Data")
all_rows = {}
for name, path in MODEL_FILES.items():
    m = load_model(path)
    all_rows[name] = get_metrics(y, m.predict(X), m.predict_proba(X)[:, 1])

comparison = pd.DataFrame(all_rows).T.round(4)
st.dataframe(comparison.style.highlight_max(axis=0, color="#c8e6c9"))