# AI-Powered Dynamic Pricing for E-commerce

This project predicts optimal selling prices for products in an e-commerce setup based on product attributes, brand, category, and ratings. The model is trained on structured data using a Random Forest Regressor and is designed to output human-readable results, suitable for business stakeholders. A Streamlit app offers real-time interaction with the model.

---

## Project Objective
- Predict accurate and competitive prices using product-level and seasonal features.
- Serve business-ready outputs through category decoding and re-scaling.
- Provide an interactive dashboard for business and product teams.

---

## Features
- Categorical feature encoding and inverse decoding
- Scaled price prediction using MinMaxScaler
- Random Forest Regressor for robust performance
- Human-readable output CSV (decoded format)
- Interactive UI for real-time prediction via Streamlit

---

## Tech Stack
- pandas – Data loading and preprocessing
- scikit-learn – Random Forest, label encoding, scaling
- matplotlib/seaborn – Exploratory data visualization
- joblib – Model and encoder serialization
- streamlit – Web UI for real-time predictions
- plotly – Optional visual enhancements in dashboard

---

## Folder Structure
```
dynamic_pricing_model/
├── data/
│   ├── train.csv
│   └── test.csv
├── dynamic_pricing.py
├── enhancements.py
├── dynamic_pricing_model.pkl
├── label_encoders.pkl
├── scaler.pkl
├── test_predictions_readable.csv
└── requirements.txt
```

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Pre123140/dynamic_pricing_predictions
cd dynamic_pricing_model
```

### 2. (Optional) Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Training and Prediction Pipeline
```bash
python dynamic_pricing.py
```

### 5. Launch the Streamlit App (Optional UI)
```bash
streamlit run enhancements.py
```

---

## Key Outputs
- test_predictions_readable.csv — Final predictions in decoded form
  - Contains readable brand, subcategory, and predicted prices
- dynamic_pricing_model.pkl — Trained Random Forest model
- scaler.pkl and label_encoders.pkl — Required for decoding in dashboard

---

## Conceptual Study
For a detailed explanation of model architecture, encoding strategy, business logic, and accuracy optimization, refer to the [Conceptual Study PDF](https://github.com/Pre123140/DYNAMIC_PRICING_MODEL/blob/main/Conceptual_Study_Dynamic_Pricing.pdf).

---

## Future Improvements
- Integrate competitor pricing as a new feature
- Test XGBoost or LightGBM for faster training
- Connect to live e-commerce systems for real-time repricing
- Add category-specific model tuning

---
## License

This project is open for educational use only. For commercial deployment, contact the author.

---

##  Contact
If you'd like to learn more or collaborate on projects or other initiatives, feel free to connect on [LinkedIn](https://www.linkedin.com/in/prerna-burande-99678a1bb/) or check out my [portfolio site](https://youtheleader.com/).
