# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pickle

# Step 2: Load the Dataset
df = pd.read_csv("train.csv")

# Step 3: Display Basic Info & First Few Rows
print(df.info())
print(df.head())

# Step 4: Handle Missing Values
df["Item_Rating"].fillna(df["Item_Rating"].mean(), inplace=True)

# Step 5: Convert 'Date' to datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Step 6: Extract Date Features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.weekday
df["DayOfYear"] = df["Date"].dt.dayofyear
df.drop(columns=["Date"], inplace=True)

# Step 7: Convert Data Types
df["Item_Rating"] = df["Item_Rating"].astype(float)

# Step 8: Label Encoding for Categorical Columns
label_encoders = {}
for col in ["Product", "Product_Brand", "Item_Category", "Subcategory_1", "Subcategory_2"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Step 9: Visualize Outliers (Optional)
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Selling_Price"])
plt.title("Selling Price Outliers")
plt.show()

# Step 10: Normalize Selling Price
scaler = MinMaxScaler()
df["Selling_Price"] = scaler.fit_transform(df[["Selling_Price"]])

# Step 11: Feature Selection
features = ["Product", "Product_Brand", "Item_Category", "Subcategory_1", "Subcategory_2",
            "Item_Rating", "Year", "Month", "Day", "DayOfWeek", "DayOfYear"]
target = "Selling_Price"
X = df[features]
y = df[target]

# Step 12: Clean Data
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
df_clean = pd.concat([X, y], axis=1).dropna()
X_clean = df_clean[features]
y_clean = df_clean[target]

# Step 13: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

print("Linear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))
print("MSE:", mean_squared_error(y_test, model.predict(X_test)))
print("R2:", r2_score(y_test, model.predict(X_test)))

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest Performance:")
print("MAE:", mean_absolute_error(y_test, rf_model.predict(X_test)))
print("MSE:", mean_squared_error(y_test, rf_model.predict(X_test)))
print("R2:", r2_score(y_test, rf_model.predict(X_test)))

# Feature Importance
importances = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

# GridSearchCV for Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Optimized Model
optimized_rf_model = RandomForestRegressor(**best_params, random_state=42)
optimized_rf_model.fit(X_train, y_train)

# Save Model
joblib.dump(optimized_rf_model, "dynamic_pricing_model.pkl")

# Save Label Encoders and Scaler
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ✅ Final Console Confirmation
print("\n✅ Model, label encoders, and scaler saved successfully.")
