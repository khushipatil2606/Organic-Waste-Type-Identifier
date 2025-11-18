import streamlit as st
import pandas as pd
import joblib

st.title("🌿 Organic Waste Type Identifier")

model = joblib.load("waste_best_model.joblib")
scaler = joblib.load("waste_scaler.joblib")
features = joblib.load("waste_features.joblib")

# FIXED: Manual class mapping (so model NEVER shows 0,1,2,3 again)
class_mapping = {
    0: "Food",
    1: "Leaves",
    2: "Other",
    3: "Paper"
}

moisture = st.number_input("Moisture (%)", 0, 100, 15)
texture = st.number_input("Texture Score", 0, 10, 7)
weight = st.number_input("Weight (g)", 0, 500, 30)
cn_ratio = st.number_input("C:N Ratio", 0, 200, 70)

df_input = pd.DataFrame([[moisture, texture, weight, cn_ratio]], columns=features)
df_scaled = scaler.transform(df_input)

if st.button("Predict Waste Type"):
    pred = model.predict(df_scaled)[0]   # This gives number 0,1,2,3

    # FIX: Convert number to class name
    waste = class_mapping[pred]

    st.success(f"♻️ Predicted Waste Type: **{waste}**")
