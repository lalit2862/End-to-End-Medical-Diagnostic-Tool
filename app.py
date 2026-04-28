import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("diabetes.csv")

# -------------------------------
# Preprocessing
# -------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Model Training
# -------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🩺 Diabetes Prediction System")

st.write("Enter patient details:")

preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")