import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set page config
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction using Linear Regression")
st.write("This app uses the Kaggle House Prices dataset (`train.csv`).")

# Load dataset
data = pd.read_csv("Linear\train.csv")
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Ensure target column exists
if 'SalePrice' not in data.columns:
    st.error("Dataset must include 'SalePrice' column as target.")
    st.stop()

# Split features and target
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Full model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
st.write(f"**R-squared (R²):** {r2:.2f}")

# Plot actual vs predicted
st.subheader("📈 Actual vs Predicted Prices")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6, color='green')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
ax.set_xlabel("Actual Sale Price")
ax.set_ylabel("Predicted Sale Price")
ax.set_title("Actual vs Predicted House Prices")
st.pyplot(fig)
