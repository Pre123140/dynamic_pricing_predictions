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

# Step 2: Load the Dataset
df = pd.read_csv("train.csv")

# Step 3: Display Basic Info & First Few Rows
print(df.info())  # Check column types and missing values
print(df.head())  # Preview first 5 rows

# Step 4: Handle Missing Values
df["Item_Rating"].fillna(df["Item_Rating"].mean(), inplace=True)

# Step 5: Convert 'Date' to datetime format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # Convert 'Date' to datetime format
print(df["Date"].head())  # Check conversion

# Step 6: Extract useful date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.weekday
df["DayOfYear"] = df["Date"].dt.dayofyear

# Drop the original 'Date' column after extracting features
df.drop(columns=["Date"], inplace=True)

# Step 7: Convert Data Types
df["Item_Rating"] = df["Item_Rating"].astype(float)  # Convert 'Item_Rating' to float

# Step 8: Handle Categorical Data Using Label Encoding
label_encoders = {}
for col in ["Product", "Product_Brand", "Item_Category", "Subcategory_1", "Subcategory_2"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to ensure correct encoding
    label_encoders[col] = le  # Store encoders for inverse transformation if needed

# Step 9: Check for Outliers in 'Selling_Price'
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Selling_Price"])
plt.show()

# Step 10: Normalize the 'Selling_Price' Column
scaler = MinMaxScaler()
df["Selling_Price"] = scaler.fit_transform(df[["Selling_Price"]])

# Step 11: Feature Selection
features = ["Product", "Product_Brand", "Item_Category", "Subcategory_1", "Subcategory_2", "Item_Rating", "Year", "Month", "Day", "DayOfWeek", "DayOfYear"]
target = "Selling_Price"

X = df[features]
y = df[target]

# Step 12: Ensure Consistency Between X and y (Check for NaN and align rows)
X = X.apply(pd.to_numeric, errors='coerce')  # Coerce any errors to NaN
y = pd.to_numeric(y, errors='coerce')  # Coerce any errors to NaN

# Drop rows where there are NaN values in either X or y
df_clean = pd.concat([X, y], axis=1).dropna()

# Split the data again after cleaning
X_clean = df_clean[features]
y_clean = df_clean[target]

# Step 13: Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Print the shapes of the datasets
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Print model coefficients
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Predict selling price using the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict selling price using the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the results
print("Random Forest Model Performance:")
print("Mean Absolute Error (MAE):", mae_rf)
print("Mean Squared Error (MSE):", mse_rf)
print("R² Score:", r2_rf)

# Get feature importance scores from the model
feature_importance = rf_model.feature_importances_

# Create a dataframe to visualize feature importance
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance for Price Prediction')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20],     # Maximum depth of each tree
    'min_samples_split': [2, 5, 10], # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]    # Minimum samples required at each leaf node
}

# Initialize the GridSearchCV
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model to training data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Train the final model with optimized parameters
optimized_rf_model = RandomForestRegressor(**best_params, random_state=42)
optimized_rf_model.fit(X_train, y_train)

# Predict selling prices using the optimized model
y_pred_optimized = optimized_rf_model.predict(X_test)

# Calculate performance metrics
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

# Print the results
print("Optimized Model Evaluation:")
print("Mean Absolute Error (MAE):", mae_optimized)
print("Mean Squared Error (MSE):", mse_optimized)
print("R² Score:", r2_optimized)

# Save the trained model to a file
joblib.dump(model, "dynamic_pricing_model.pkl")

# Load the trained model
model = joblib.load("dynamic_pricing_model.pkl")

# Load the test dataset
test_data = pd.read_csv("test.csv")

# Display test data sample
print("Test Data Sample:")
print(test_data.head())

# Ensure all categorical features are encoded properly
for col in label_encoders.keys():
    if col in test_data.columns:
        # Map known labels, assign -1 to unknown labels
        test_data[col] = test_data[col].apply(lambda x: label_encoders[col].classes_.tolist().index(x) if x in label_encoders[col].classes_ else -1)

# Ensure all columns in test data match the training feature set
missing_cols = set(X_train.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0  # Add missing columns with a default value

# Align test data column order to match training data
test_data = test_data[X_train.columns]

# Convert numerical columns to correct dtype to avoid dtype promotion errors
test_data = test_data.astype(float)

# Predict selling prices using the trained model
test_data["Predicted_Selling_Price"] = model.predict(test_data)

# Print sample predictions
print("\nSample Predictions:")
print(test_data[["Product", "Product_Brand", "Predicted_Selling_Price"]].head())

# After making predictions and getting the predicted results, reverse the label encoding
def inverse_transform_safe(column_name, column_data, label_encoders):
    """
    Safely reverse the encoding of a column, handling unseen labels.
    """
    try:
        return label_encoders[column_name].inverse_transform(column_data.astype(int))
    except ValueError as e:
        print(f"Warning: Unseen labels encountered in column '{column_name}'.")
        return column_data  # Return as is or map to a default value if needed.

# Reverse encoding for categorical columns, with error handling
test_data["Product"] = inverse_transform_safe("Product", test_data["Product"], label_encoders)
test_data["Product_Brand"] = inverse_transform_safe("Product_Brand", test_data["Product_Brand"], label_encoders)
test_data["Item_Category"] = inverse_transform_safe("Item_Category", test_data["Item_Category"], label_encoders)
test_data["Subcategory_1"] = inverse_transform_safe("Subcategory_1", test_data["Subcategory_1"], label_encoders)
test_data["Subcategory_2"] = inverse_transform_safe("Subcategory_2", test_data["Subcategory_2"], label_encoders)

# Now, your test_data should have the original categorical labels back (with handled unseen labels)
print("\nUpdated Test Data with Predicted Selling Prices:")
print(test_data[["Product", "Product_Brand", "Item_Category", "Subcategory_1", "Subcategory_2", "Predicted_Selling_Price"]].head())

# Save the updated predictions to a CSV
test_data.to_csv("test_predictions_readable.csv", index=False)

print("\n✅ Predictions saved successfully to 'test_predictions_readable.csv'!")
