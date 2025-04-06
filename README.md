# 🛒 Dynamic Pricing Model for E-commerce

An AI-powered tool that predicts optimal product prices based on brand, category, customer rating, and seasonal demand. Designed for fast-changing markets and built using Random Forest Regression, this model adapts to real-world pricing strategies and ensures interpretability through category decoding and price re-scaling.

---

## 🚀 Project Overview

This machine learning solution enables:
- Predictive pricing using structured product + temporal data
- Feature importance insights for data-driven decisions
- Real-time inference through Streamlit UI
- Human-readable price predictions for business users

---

## 📁 Folder Structure
dynamic_pricing_model/
├── data/
│   ├── train.csv                     # Training dataset
│   └── test.csv                      # Test dataset
├── dynamic_pricing.py               # Full preprocessing, training, and prediction pipeline
├── dynamic_pricing_model.pkl        # Final trained model
├── label_encoders.pkl               # Saved encoders for inverse transformation
├── scaler.pkl                       # MinMaxScaler for rescaling prices
├── test_predictions_readable.csv    # Decoded, final price predictions
├── enhancements.py                  # Streamlit app for enhancements & UI
└── README.md                        # Project summary and usage guide







---

## 🧠 Core Features

- 🔄 **Dynamic Model**: Learns from brand, subcategory, ratings, and calendar behavior
- 🧪 **Test-Time Prediction**: Works on unseen data with fallback encoding
- 🔍 **Inverse Decoding**: Translates predictions into human-readable format
- 📉 **MinMax Scaling**: Ensures numerical stability in predictions
- 📊 **Feature Importance**: Helps identify top factors affecting price

---

## 🛠️ Tools & Libraries

| Tool/Library      | Purpose                            |
|------------------|------------------------------------|
| Python, pandas    | Data manipulation and prep         |
| scikit-learn      | ML models, scaling, evaluation     |
| matplotlib, seaborn | Visualization                   |
| joblib, pickle    | Model and encoder persistence      |
| streamlit, plotly | Interactive UI (optional)         |

---

## 💻 How to Run

1. **Install Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib plotly




Train Model & Predict Prices
python dynamic_pricing.py


Launch Interactive Streamlit Dashboard
streamlit run enhancements.py


Output
Final results saved to:
test_predictions_readable.csv

Includes:
Product names and brand labels
Category information
Final predicted selling prices (decoded)


Possible Extensions
Add competitor pricing as an external feature

Deploy via API or integrate with e-commerce CMS

Use XGBoost or LightGBM for improved speed/performance

Connect to real-time sales streams for live pricing

⚠️ Disclaimer
This project is for educational and illustrative purposes only. It uses synthetic and anonymized datasets. It is not intended for production use without further validation and adaptation.

📎 License
MIT License — Free for personal and academic use.


Let me know if you want the Medium and LinkedIn blog posts next or want help setting up the repo structure or `.gitignore`.


