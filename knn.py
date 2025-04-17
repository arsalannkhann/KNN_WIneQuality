import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
KNN = joblib.load('Streamlit/knn_model.pkl')
scaler = joblib.load('Streamlit/scaler.pkl')

# Load the dataset
data = pd.read_csv('/Users/arsalankhan/.cache/kagglehub/datasets/yasserh/wine-quality-dataset/versions/1/WineQT.csv')

# Streamlit app
st.title("Wine Quality Prediction App")

st.sidebar.header("Input Features")
st.sidebar.write("Adjust the sliders to input feature values.")

# Sidebar sliders for input features
fixed_acidity = st.sidebar.slider("Fixed Acidity", float(data['fixed acidity'].min()), float(data['fixed acidity'].max()), float(data['fixed acidity'].mean()))
volatile_acidity = st.sidebar.slider("Volatile Acidity", float(data['volatile acidity'].min()), float(data['volatile acidity'].max()), float(data['volatile acidity'].mean()))
citric_acid = st.sidebar.slider("Citric Acid", float(data['citric acid'].min()), float(data['citric acid'].max()), float(data['citric acid'].mean()))
residual_sugar = st.sidebar.slider("Residual Sugar", float(data['residual sugar'].min()), float(data['residual sugar'].max()), float(data['residual sugar'].mean()))
chlorides = st.sidebar.slider("Chlorides", float(data['chlorides'].min()), float(data['chlorides'].max()), float(data['chlorides'].mean()))
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", float(data['free sulfur dioxide'].min()), float(data['free sulfur dioxide'].max()), float(data['free sulfur dioxide'].mean()))
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", float(data['total sulfur dioxide'].min()), float(data['total sulfur dioxide'].max()), float(data['total sulfur dioxide'].mean()))
density = st.sidebar.slider("Density", float(data['density'].min()), float(data['density'].max()), float(data['density'].mean()))
pH = st.sidebar.slider("pH", float(data['pH'].min()), float(data['pH'].max()), float(data['pH'].mean()))
sulphates = st.sidebar.slider("Sulphates", float(data['sulphates'].min()), float(data['sulphates'].max()), float(data['sulphates'].mean()))
alcohol = st.sidebar.slider("Alcohol", float(data['alcohol'].min()), float(data['alcohol'].max()), float(data['alcohol'].mean()))

# Create a dataframe for the input features
input_data = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free sulfur dioxide': [free_sulfur_dioxide],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

# Standardize the input features
scaled_input = scaler.transform(input_data)

# Predict the wine quality
prediction = KNN.predict(scaled_input)
# Add a button for prediction
if st.button("Predict"):
    # Display the prediction
    st.subheader("Prediction")
    st.write(f"The predicted wine quality is: {prediction[0]}")



# Add a button to show the dataset
if st.button("Show Dataset"):
    st.subheader("Dataset")
    st.write(data)

