import os
import sys
import streamlit as st
import pandas as pd
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

model_path = os.path.join("models", "random_forest.pkl")
# Load trained model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)

# Page title
st.title("💼 Term Deposit Subscription Predictor")
st.markdown("Predict whether a bank client will subscribe to a term deposit based on marketing data.")

# Input form
with st.form("subscription_form"):
    st.subheader("📋 Client Information")

    age = st.number_input("Age", min_value=18, max_value=100, step=1)

    job = st.selectbox("Job", ["admin.", "unknown", "unemployed", "management", "housemaid",
                               "entrepreneur", "student", "blue-collar", "self-employed",
                               "retired", "technician", "services"])

    marital = st.selectbox("Marital Status", ["married", "divorced", "single"])

    education = st.selectbox("Education Level", ["unknown", "secondary", "primary", "tertiary"])

    default = st.radio("Has Credit in Default?", ["no", "yes"])

    balance = st.number_input("Average Yearly Balance (in Euros)", step=1)

    housing = st.radio("Has Housing Loan?", ["no", "yes"])

    loan = st.radio("Has Personal Loan?", ["no", "yes"])

    contact = st.selectbox("Contact Communication Type", ["unknown", "telephone", "cellular"])

    day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, step=1)

    month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun",
                                                "jul", "aug", "sep", "oct", "nov", "dec"])

    duration = st.number_input("Last Contact Duration (in seconds)", step=1)

    campaign = st.number_input("Number of Contacts in this Campaign", step=1)

    pdays = st.number_input("Days Since Last Contact (use -1 if never contacted)", step=1)

    previous = st.number_input("Number of Previous Contacts", step=1)

    poutcome = st.selectbox("Outcome of Previous Campaign", ["unknown", "other", "failure", "success"])

    submit = st.form_submit_button("🔍 Predict")

# Inference
if submit:
    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure encoding matches training (must match training pipeline)
    # Here assuming the model was trained on label encoded data
    # You may need to apply same LabelEncoders or use OneHotEncoder consistently
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.success(f"🧠 Prediction: **{'Yes' if prediction == 1 else 'No'}**")
        st.info(f"📊 Probability of Subscription: **{probability * 100:.2f}%**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        st.markdown
