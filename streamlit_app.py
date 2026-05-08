import streamlit as st
import pandas as pd
import pickle

# 1. Load your model and scaler
model = pickle.load(open('xgboost_diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Diabetes Risk Prediction System")
st.write("Enter the patient details below to predict diabetes risk.")

# 2. Create Input Fields (Match these to your model's features)
pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose", 0, 300, 100)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
# ... Add other fields like BMI, Age, etc.

if st.button("Predict"):
    # 3. Prepare data for prediction
    input_data = [[pregnancies, glucose, blood_pressure, ...]] # Add all inputs here
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")