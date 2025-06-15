# app/predict.py

import streamlit as st
import pandas as pd
import joblib
import os
import sys
import importlib.util

# =======================
# Dynamic import utility
# =======================
def import_module_from_src(filename, name=None):
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', filename))
    spec = importlib.util.spec_from_file_location(name or filename, full_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name or filename] = module
    spec.loader.exec_module(module)
    return module

# ======================
# Import Feature Functions
# ======================
feature_mod = import_module_from_src('feature_engineering.py', 'feature_engineering')
engineer_features = feature_mod.engineer_features
encode_features = feature_mod.encode_features

# ======================
# Load Model
# ======================
model_path = os.path.join("models", "random_forest.pkl")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)

# =======================
# Streamlit UI
# =======================
st.title("💼 Term Deposit Subscription Predictor")
st.markdown("Predict whether a bank client will subscribe to a term deposit based on marketing data.")

with st.form("subscription_form"):
    st.subheader("📋 Client Information")

    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    job = st.selectbox("Job", ["admin.", "unknown", "unemployed", "management", "housemaid",
                               "entrepreneur", "student", "blue-collar", "self-employed",
                               "retired", "technician", "services"])
    marital = st.selectbox("Marital Status", ["married", "divorced", "single"])
    education = st.selectbox("Education Level", ["unknown", "secondary", "primary", "tertiary"])
    default = st.radio("Has Credit in Default?", ["no", "yes"])
    #balance = st.number_input("Average Yearly Balance (in Euros)", step=1)
    housing = st.radio("Has Housing Loan?", ["no", "yes"])
    loan = st.radio("Has Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Communication Type", ["unknown", "telephone", "cellular"])
    day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, step=1)
    month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun",
                                                "jul", "aug", "sep", "oct", "nov", "dec"])
    duration = st.number_input("Last Contact Duration (in seconds)", step=1)
    campaign = st.number_input("Number of Contacts in this Campaign", step=1)
    pdays = st.number_input("Days Since Last Contact (-1 if never contacted)", step=1)
    previous = st.number_input("Number of Previous Contacts", step=1)
    poutcome = st.selectbox("Outcome of Previous Campaign", ["unknown", "other", "failure", "success"])

    submit = st.form_submit_button("🔍 Predict")

# =======================
# Inference
# =======================
if submit:
    # Prepare input
    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        #'balance': balance,
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

    input_df = pd.DataFrame([input_dict])

    try:
        # Apply same pipeline used in training
        engineered = engineer_features(input_df)
        encoded = encode_features(engineered)

        prediction = model.predict(encoded)[0]
        probability = model.predict_proba(encoded)[0][1]

        st.success(f"🧠 Prediction: **{'Yes' if prediction == 1 else 'No'}**")
        st.info(f"📊 Probability of Subscription: **{probability * 100:.2f}%**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
