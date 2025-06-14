# 📈 Term Deposit Subscription Prediction
As part of our data analytics initiative, this project aims to develop a predictive model to assist the marketing team in identifying clients who are likely to subscribe to a bank term deposit. Using historical campaign data, we apply machine learning techniques to build, evaluate, and deploy a model for actionable insights.

# 🎯 Objective
Predict whether a client will subscribe to a term deposit (y = "yes" or "no") based on various client and campaign-related features.

# 🧠 Project Workflow
1. 🔍 Exploratory Data Analysis (EDA)
Analyzed distributions, correlations, and patterns across features.

Handled missing values and outliers where applicable.

Normalized and transformed data for compatibility with ML models.

2. 🏗️ Feature Engineering
Encoded categorical variables using one-hot encoding.

Created derived features to enhance predictive power.

Identified top features contributing to target variable.

3. 🤖 Model Building
Trained multiple classification models:

Logistic Regression

Random Forest

XGBoost

Final model selection based on ROC-AUC score.

4. 📊 Model Evaluation
Evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Addressed class imbalance using stratified sampling and evaluation metrics sensitive to imbalance.

# 5. 📣 Insights & Recommendations
Identified key features influencing client subscription.

Recommended client segments to prioritize in future marketing campaigns.

Final model made available for real-time inference via Streamlit app.

📂 Project Structure
bash
Copy
Edit
.
├── app/                    # Streamlit UI for interactive predictions
│   └── predict.py
├── data/
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned and feature-engineered data
├── models/                 # Saved trained models (only best is kept)
│   └── best_model.pkl
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── reports/                # Summary reports and visualizations
├── src/                    # Scripts (e.g., feature_engineering.py)
│   ├── feature_engineering.py
│   └── train_models.py
├── requirements.txt        # Required packages
└── README.md               # Project documentation
🚀 Running the Streamlit App
To interact with the trained model:

## 1. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
## 2. Run the app
bash
Copy
Edit
streamlit run app/predict.py
This will launch a web interface to input client data and receive a real-time prediction.

# 🧾 Data Description
The dataset contains data from direct phone-based marketing campaigns of a Portuguese banking institution. The campaigns aimed to promote term deposits to clients.

Available Files:
bank-additional-full.csv — Main dataset with 41,188 examples and 20 input variables.

bank-additional.csv — 10% subset (4,119 examples).

bank-full.csv — Full older version (17 input features).

bank.csv — 10% of the older version.

Dataset Source: UCI Machine Learning Repository – Bank Marketing Dataset

# 📈 Final Model Performance (Example)
Metric	Score
Accuracy	0.91
Precision	0.82
Recall	0.72
F1 Score	0.77
ROC-AUC	0.89

Performance is based on the best model selected from the training pipeline.

✅ Deliverables
✅ Clean and structured source code

✅ Trained and serialized best model

✅ Streamlit web app for live prediction

✅ Jupyter notebooks with EDA and modeling process

✅ Summary report of findings and insights

# 💬 Interview Notice
You may be asked to demonstrate the model during an interview. Ensure the environment is set up and the Streamlit app is running locally:

bash
Copy
Edit
streamlit run app/predict.py
# 📌 Contact
For questions or suggestions, feel free to open an Issue or Pull Request.

