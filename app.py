import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading Dataset
df = pd.read_csv("data/dataset.csv")

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

# Visualization: Actual vs Predicted Prices
st.subheader("📈 Actual vs Predicted Prices")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color="blue", alpha=0.6)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed")
ax.set_xlabel("Actual Prices (Lakh ₹)")
ax.set_ylabel("Predicted Prices (Lakh ₹)")
ax.set_title("Actual vs Predicted House Prices")
st.pyplot(fig)

# House Price Prediction Section
st.subheader("🏠 Predict House Price")

# User inputs for prediction
size_input = st.slider("Size (sq ft)", min_value=800, max_value=3500, step=100)
bedrooms_input = st.slider("Number of Bedrooms", min_value=1, max_value=5, step=1)
bathrooms_input = st.slider("Number of Bathrooms", min_value=1, max_value=4, step=1)
distance_input = st.slider("Distance from City (km)", min_value=1, max_value=20, step=1)
age_input = st.slider("House Age (years)", min_value=0, max_value=50, step=1)


# Predict button
if st.button("Predict Price"):
    user_data = np.array([[size_input, bedrooms_input, bathrooms_input, distance_input, age_input]])
    predicted_price = model.predict(user_data)[0]
    st.success(f"🏡 Estimated House Price: ₹ {predicted_price:.2f} Lakh")

st.subheader("1️⃣ Assumptions of Linear Regression (Very Important!)")
st.write('''Linear Regression makes some assumptions. If they don’t hold, your model may not work well.
- Linearity → Relationship between features & target must be linear.
- Independence → No correlation between residuals (errors).
- Homoscedasticity → Variance of residuals should be constant across predictions.
- Normality of Residuals → Residuals should be normally distributed.
- No Multicollinearity → Features should not be highly correlated.''')

st.subheader("2️⃣ Feature Selection & Engineering")
st.write('''- Removing Irrelevant Features → Use p-values from statsmodels or Recursive Feature Elimination (RFE).
- Handling Categorical Variables → Use One-Hot Encoding or Label Encoding.
- Scaling Data → Standardization (Z-score) or MinMax Scaling.
''')

st.subheader("3️⃣ Handling Outliers & Influential Points")
st.write('''- Outliers → Use IQR (Interquartile Range) or Z-score to detect and remove them.
- Influential Points → Use Cook’s Distance to find and handle them.''')

st.subheader("4️⃣ Regularization (Lasso & Ridge Regression)")
st.write('''If your model overfits, try L1 (Lasso) or L2 (Ridge) regularization to reduce complexity.
- Lasso Regression (L1) → Shrinks some coefficients to 0, good for feature selection.
- Ridge Regression (L2) → Reduces all coefficients but doesn’t make them zero.
- ElasticNet → Combination of L1 & L2.''')

st.subheader("5️⃣ Polynomial Regression (For Non-Linear Data)")
st.write('''If data isn’t perfectly linear, transform features:
- Instead of predicting Y = mX + b, try Y = aX² + bX + c
- Use PolynomialFeatures() from sklearn''')

st.subheader("6️⃣ Cross-Validation & Model Selection")
st.write('''- Use K-Fold Cross Validation to evaluate stability.
- Compare with other models like Decision Trees & Random Forest.''')