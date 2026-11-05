# Online Shoppers Purchase Prediction – ML Assignment 2

## a. Problem Statement
Online stores receive many visitors, but only a small share of sessions end in a purchase.
The goal of this project is to predict whether an online shopping session will end in a purchase
(`Revenue` = 1) or not (`Revenue` = 0), using the visitor's browsing behaviour and session details.
Five classification models are implemented on the same dataset and compared using six evaluation
metrics, and the models are demonstrated through an interactive Streamlit web app.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository – Online Shoppers Purchasing Intention Dataset
- **Instances:** 12,330 sessions (12,205 after removing 125 duplicate rows)
- **Features:** 17 input features (10 numeric, 7 categorical/binary)
- **Target:** `Revenue` (1 = purchase, 0 = no purchase)
- **Class balance:** ~15.6% purchase, ~84.4% no purchase (imbalanced dataset)
- **Missing values:** none

**Feature groups:**
- **Page activity:** `Administrative`, `Informational`, `ProductRelated` – number of pages of each type visited, with their `_Duration` columns showing the time spent on them.
- **Google Analytics metrics:** `BounceRates`, `ExitRates`, `PageValues` (average value of pages visited before a purchase) and `SpecialDay` (closeness of the visit to a special day such as Valentine's Day).
- **Session details:** `Month`, `VisitorType` (new / returning / other), `Weekend`, and coded categories `OperatingSystems`, `Browser`, `Region`, `TrafficType`.

**Preprocessing:**
- Removed 125 duplicate rows
- Converted `Weekend` and `Revenue` from True/False to 1/0
- One-hot encoding for `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`
- StandardScaler for the 10 numeric features
- 80/20 stratified train-test split (9,764 train / 2,441 test rows), keeping the same purchase ratio in both sets
- Preprocessing and model are combined in a scikit-learn `Pipeline`, so the saved model can be applied directly to raw test data in the app

## c. GitHub Repository Link
https://github.com/jyotishachaurasia1/Online-Shoppers

## Live Streamlit App:
https://jyotisha-online-shoppers.streamlit.app/

## d. Models Used and Evaluation Metrics
Models implemented: Logistic Regression, Decision Tree Classifier, K-Nearest Neighbors,
Gaussian Naive Bayes and Support Vector Machine (RBF kernel).
All metrics are calculated on the held-out test set (2,441 sessions, 382 buyers).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8812 | 0.8954 | 0.7347 | 0.3770 | 0.4983 | 0.4702 |
| Decision Tree | 0.9009 | 0.9185 | 0.7096 | 0.6204 | 0.6620 | 0.6061 |
| KNN | 0.8705 | 0.8063 | 0.6486 | 0.3770 | 0.4768 | 0.4285 |
| Naive Bayes | 0.2536 | 0.5990 | 0.1709 | 0.9791 | 0.2911 | 0.1169 |
| Support Vector Machine | 0.8947 | 0.8954 | 0.7376 | 0.5079 | 0.6016 | 0.5558 |

### Observations on Model Performance
Since only ~15.6% of test sessions are purchases, a model that always predicts "no purchase"
would already get 84.4% accuracy. So MCC and F1 are used as the main comparison metrics.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | High AUC (0.8954) and precision (0.7347) but low recall (0.3770): it found only 144 of 382 buyers. Its linear decision boundary is too simple for this data, and the class imbalance pushes it towards predicting "no purchase". |
| Decision Tree | Best model, with the highest Accuracy (0.9009), AUC (0.9185), Recall (0.6204), F1 (0.6620) and MCC (0.6061). It learns non-linear if-then rules (e.g. high PageValues with low ExitRates), catching 93 more buyers than Logistic Regression. max_depth=6 helped avoid overfitting. |
| KNN | Weakest AUC (0.8063) among the reasonable models, with low recall (0.3770). Most neighbours of any visitor are non-buyers, so the vote favours "no purchase", and the many one-hot columns make distance less meaningful. |
| Naive Bayes | Worst model (Accuracy 0.2536, MCC 0.1169). It predicted "purchase" for almost every session (Recall 0.9791, Precision 0.1709). Its assumptions of independent, normally distributed features fail here, as many features are correlated and highly skewed. |
| Support Vector Machine | Second best (MCC 0.5558, F1 0.6016) with the highest precision (0.7376). The RBF kernel captures non-linear patterns better than Logistic Regression, but it still misses about half of the buyers and is the slowest model to train. |
| **Overall Winner** | **Decision Tree**: it achieved the highest MCC and F1, the fairest metrics for imbalanced data, and gives the best balance between catching buyers and avoiding false alarms. |

## Streamlit App Features
- Upload test data as a CSV file
- Dropdown to select any of the 5 trained models
- Display of all 6 evaluation metrics for the selected model
- Confusion matrix heatmap and classification report
- Comparison table of all 5 models on the uploaded data
- The app only loads the saved models; it does not retrain on uploaded data

## Repository Structure
```
├── app.py                   # Streamlit web app
├── model_training.ipynb     # Data preparation, training and evaluation
├── explore.ipynb            # Initial data exploration
├── model/                   # Saved trained pipelines (.pkl) for all 5 models
├── test_data.csv            # Test data (20% split) with target column
├── requirements.txt         # Python dependencies
└── README.md
```

## How to Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```
Then upload `test_data.csv` from the sidebar.