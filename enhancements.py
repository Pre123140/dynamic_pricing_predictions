import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import pickle

st.set_page_config(layout="wide")
st.title("📈 Dynamic Pricing Model – Enhancements")

# Load trained model
try:
    model = joblib.load("dynamic_pricing_model.pkl")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# Load test data
try:
    test_data_raw = pd.read_csv("Test.csv")
    test_data = test_data_raw.copy()
    st.success("✅ Test data loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load test dataset: {e}")
    st.stop()

# Load encoders and scaler
try:
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    st.success("✅ Encoders and scaler loaded!")
except Exception as e:
    st.error(f"❌ Failed to load encoders or scaler: {e}")
    label_encoders = {}
    scaler = None
    st.stop()

# Safe encoding function with fallback to -1
def safe_encode(col, encoder):
    return col.apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

# Encode categorical columns
for col in label_encoders:
    if col in test_data.columns:
        test_data[col] = safe_encode(test_data[col].astype(str), label_encoders[col])

# Feature Engineering: robust handling of Date parsing
test_data["Date"] = pd.to_datetime(test_data["Date"], errors="coerce")
test_data["Year"] = test_data["Date"].dt.year.fillna(0).astype(int)
test_data["Month"] = test_data["Date"].dt.month.fillna(0).astype(int)
test_data["Day"] = test_data["Date"].dt.day.fillna(0).astype(int)
test_data["DayOfWeek"] = test_data["Date"].dt.weekday.fillna(0).astype(int)
test_data["DayOfYear"] = test_data["Date"].dt.dayofyear.fillna(0).astype(int)
test_data.drop(columns=["Date"], inplace=True)

# Fill numeric missing values
test_data["Item_Rating"] = test_data["Item_Rating"].fillna(test_data["Item_Rating"].mean())

# Align columns
features = [
    "Product", "Product_Brand", "Item_Category", "Subcategory_1", "Subcategory_2",
    "Item_Rating", "Year", "Month", "Day", "DayOfWeek", "DayOfYear"
]
for col in features:
    if col not in test_data.columns:
        test_data[col] = 0

test_data = test_data[features].astype(float)

# Predict normalized price
test_data["Predicted_Selling_Price"] = model.predict(test_data)

# Inverse scale to get actual price
if scaler:
    try:
        test_data["Predicted_Selling_Price"] = scaler.inverse_transform(
            test_data[["Predicted_Selling_Price"]]
        ).round(2)
    except Exception as e:
        st.warning(f"⚠️ Could not inverse scale prices: {e}")
else:
    st.warning("⚠️ Scaler not found – prices are still normalized.")

# Decode categorical columns
def safe_decode(col_name, encoded_col, raw_col=None):
    decoded = []
    le = label_encoders.get(col_name)
    for i, val in enumerate(encoded_col):
        try:
            decoded_val = le.inverse_transform([int(val)])[0]
        except:
            decoded_val = raw_col.iloc[i] if raw_col is not None else f"Unknown_{col_name}"
        decoded.append(decoded_val)
    return decoded

for col in ["Product", "Product_Brand", "Item_Category", "Subcategory_1", "Subcategory_2"]:
    raw_col = test_data_raw[col] if col in test_data_raw.columns else None
    test_data[col] = safe_decode(col, test_data[col], raw_col)

# Show results
st.subheader("🔍 Sample of Readable Predictions")
st.markdown("These are the final predictions with original category labels restored:")
st.dataframe(test_data[[
    "Product", "Product_Brand", "Item_Category", "Subcategory_1", "Subcategory_2", "Predicted_Selling_Price"
]].head(10))

# Optional download
st.download_button("📥 Download Readable Predictions",
                   test_data.to_csv(index=False),
                   file_name="test_predictions_readable.csv")
