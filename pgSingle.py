import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib  # To load saved ML models
import os

if os.path.exists("phase1_single.py"):
        with open("phase1_single.py") as f:
            exec(f.read())

# Load trained XGBoost models from Phase 1
trained_models = joblib.load("trained_models.pkl")  # Ensure this file exists

# Set page title
st.set_page_config(page_title="Single Job", layout="centered")

if st.button("Return to Home Page"):
    st.switch_page("pgTitle.py")

# Title
st.markdown("## Enter Job Information:")

left, middle, right = st.columns([3, 5, 8])

with left:
    orderQty = st.number_input("Order Qty", min_value=0)
with middle:
    job1 = st.selectbox("Select Machine 1", ["", "Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"])
    job2 = st.selectbox("Select Machine 2", ["", "Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"])
    job3 = st.selectbox("Select Machine 3", ["", "Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"])
    job4 = st.selectbox("Select Machine 4", ["", "Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"])
    job5 = st.selectbox("Select Machine 5", ["", "Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"])

with right:
    col1, col2 = st.columns(2)
    with col1:
        ftOFFSET = st.selectbox("OFFSET", ["", "YES", "NO"])
        ftFLUTECODE = st.selectbox("FLUTE CODE", ["", "0", "B", "BC", "C", "E", "EB", "EC", "F", "SBS", "STRN", "X"])
        ftCLOSURE = st.selectbox("CLOSURE TYPE", ["", "0"])
        ftCOMPONENT = st.selectbox("COMPONENT CODE", ["", "0", "10PT", "12PT", "16PT", "18PT", "20PT"])
        ftROTARY = st.selectbox("ROTARY DC", ["", "YES", "NO"])
    with col2:
        ftTESTCODE = st.number_input("TEST CODE", min_value=0, max_value=999, value=None)
        ftNUMBERUP = st.number_input("NUMBER UP ENTRY", min_value=0, max_value=100, value=None)
        ftBLANKWIDTH = st.number_input("BLANK WIDTH", min_value=0, value=None)
        ftBLANKLENGTH = st.number_input("BLANK LENGTH", min_value=0, value=None)
        ftITEMWIDTH = st.number_input("ITEM WIDTH", min_value=0, value=None)
        ftITEMLENGTH = st.number_input("ITEM LENGTH", min_value=0, value=None)

st.markdown("---")  

# Dictionary of user inputs
user_inputs = {
    "Machine Group 1": job1,
    "Machine Group 2": job2,
    "Machine Group 3": job3,
    "Machine Group 4": job4,
    "Machine Group 5": job5,
    "OFFSET?": ftOFFSET,
    "Flute Code Grouped": ftFLUTECODE,
    "Closure Type": ftCLOSURE,
    "Component Code Grouped": ftCOMPONENT,
    "Rotary DC": ftROTARY,
    "Test Code": ftTESTCODE,
    "Number Up Entry": ftNUMBERUP,
    "Blank Width": ftBLANKWIDTH,
    "Blank Length": ftBLANKLENGTH,
    "Item Width": ftITEMWIDTH,
    "Item Length": ftITEMLENGTH
}

# Remove empty inputs
user_inputs = {k: v for k, v in user_inputs.items() if v not in [None, "", " "]}

# Convert into a DataFrame
df_user = pd.DataFrame([user_inputs])

# One-hot encoding
df_encoded = pd.get_dummies(df_user)
df_encoded = df_encoded.reindex(columns=trained_models[0].feature_names_in_, fill_value=0)

# Predict using all models
all_preds = np.array([model.predict(df_encoded) for model in trained_models]).T
pred_mean = all_preds.mean(axis=1)[0]
pred_std = all_preds.std(axis=1)[0]

# Display results
st.write(f"**Predicted Waste %:** {round(pred_mean, 2)} ± {round(pred_std, 2)}")

# Adjust order quantities
wasteQty = round(pred_mean * orderQty)
optQty = round(orderQty - wasteQty)
finalQty = round(orderQty + (0.001 * orderQty))

st.write("### Optimal Starting Quantities")
st.write(f"Start with: {optQty}")
st.write(f"Estimated Waste: {wasteQty}")
st.write(f"Final Output Quantity: {finalQty}")
