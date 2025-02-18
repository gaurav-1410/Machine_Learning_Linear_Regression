import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading Dataset
df = pd.read_csv("./data/dataset.csv")

# Streamlit UI
st.title("🏡 House Price Prediction using Multiple Linear Regression")

# Dispaling Dataset
st.subheader("Dataset Overview")
st.write("This dataset contains information about houses and their respective prices.")
st.dataframe(df.head(10))

# Splitting Dataset
X = df.drop(columns= ['Price_lakh']) # Features or Independent Variable
y = df['Price_lakh'] # Target Varible or Dependent Variable

# Train-Test-Split (using 80% for training and 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Training Multiple Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction 
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Displaying Result
st.subheader("📊 Model Evaluation Metrics")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")
