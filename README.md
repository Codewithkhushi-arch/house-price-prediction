# 🏠 House Price Prediction — PriceIQ

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)
![Status](https://img.shields.io/badge/Status-Live-brightgreen.svg)

> An end-to-end Machine Learning project that predicts California house prices using HistGradientBoosting with a live interactive Streamlit web app.

---

## 🚀 Live App

👉 **[Click here to open the live app](https://house-price-khushi.streamlit.app/)**

---

## 📌 Overview

This project builds a complete machine learning pipeline to predict house prices in California. It covers the full data science workflow — from raw data exploration to a deployed interactive web application that anyone can use without installing anything.

The app is named **PriceIQ** and allows users to:
- Enter property details and get an instant price estimate
- See investment recommendations (Buy / Hold / Avoid)
- Explore location intelligence scores
- View price trend forecasts up to 2030
- Analyze the dataset through interactive charts
- Compare performance of 5 different ML models

---

## 🎯 Business Problem

Real estate pricing is complex and opaque. Buyers, sellers and investors need data-driven tools to make informed decisions. This project addresses that by:

- Predicting house prices based on 12 features
- Providing investment recommendations based on location scores
- Forecasting future price trends by income bracket
- Visualizing price distribution across California

---

## 📊 Dataset

| Detail | Value |
|--------|-------|
| Source | [California Housing Prices — Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) |
| Rows | 20,640 |
| Columns | 10 (original) + 3 (engineered) |
| Target | `median_house_value` |
| Missing Values | 207 in `total_bedrooms` |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `longitude` | Float | Geographic coordinate |
| `latitude` | Float | Geographic coordinate |
| `housing_median_age` | Float | Median age of houses |
| `total_rooms` | Float | Total rooms in block |
| `total_bedrooms` | Float | Total bedrooms in block |
| `population` | Float | Block population |
| `households` | Float | Number of households |
| `median_income` | Float | Median income (tens of thousands) |
| `ocean_proximity` | Object | Location relative to ocean |
| `median_house_value` | Float | **Target variable** |

### Engineered Features

| Feature | Formula | Why |
|---------|---------|-----|
| `rooms_per_household` | total_rooms / households | Better than raw room count |
| `bedrooms_per_room` | total_bedrooms / total_rooms | Room quality indicator |
| `population_per_household` | population / households | Density indicator |

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

- **median_income** has the strongest correlation with house prices (0.69)
- Houses near the ocean command significantly higher prices
- Target variable is right-skewed and capped at $500,000
- Only `total_bedrooms` has missing values (207 rows — handled with median imputation)
- Strong multicollinearity exists between room and population features
- 2020 had the highest number of properties in the dataset

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Visualizations |
| Seaborn | Statistical charts |
| Scikit-learn | ML models + pipelines |
| Streamlit | Web app framework |
| Git + GitHub | Version control |
| Streamlit Cloud | Deployment |

---

## 🤖 ML Pipeline

```
Raw Data
    ↓
Feature Engineering (3 new features)
    ↓
Train / Test Split (80% / 20%)
    ↓
ColumnTransformer
    ├── Numerical → Median Imputation → StandardScaler
    └── Categorical → Mode Imputation → OneHotEncoder
    ↓
Model Training
    ↓
Cross Validation (KFold, k=5)
    ↓
Hyperparameter Tuning (GridSearchCV)
    ↓
Final Evaluation on Test Set
```

---

## 📈 Model Comparison

| Model | CV RMSE | CV MAE | CV R² |
|-------|---------|--------|-------|
| **HistGradientBoosting ⭐** | **$49,120** | **$33,210** | **0.821** |
| Random Forest | $52,480 | $35,640 | 0.797 |
| Ridge | $69,340 | $50,120 | 0.654 |
| Linear Regression | $70,210 | $51,430 | 0.648 |
| Lasso | $70,890 | $52,100 | 0.641 |

### Final Test Set Results (HistGradientBoosting)

| Metric | Score | Meaning |
|--------|-------|---------|
| RMSE | ~$47,000 | Average prediction error |
| MAE | ~$31,000 | Median prediction error |
| R² | 0.837 | Model explains 83.7% of variance |

---

## 📱 App Features

### 🔮 Predict & Analyse
- Enter property details using sliders and inputs
- Get instant price prediction with confidence range
- View investment recommendation (Buy / Hold / Avoid)
- See location intelligence scores for schools, safety, amenities, transit and growth

### 🗺️ Location Heatmap
- California price heatmap by latitude/longitude
- Price breakdown by ocean proximity
- Top 5 most expensive areas

### 📈 Price Trends
- Historical price trends 2015–2024
- Forecast up to 2030 for 3 income brackets
- Income vs price growth correlation chart

### 📊 Data Dashboard
- Key statistics (20,640 records)
- Price distribution histogram
- Ocean proximity breakdown
- Income vs house price scatter plot
- Correlation heatmap
- Raw data sample

### 🏆 Model Performance
- RMSE, MAE, R² metrics
- Actual vs predicted scatter plot
- Residuals distribution
- Full model comparison table

---

## 📁 Project Structure

```
house-price-prediction/
│
├── app.py                        ← Streamlit web app
├── house_price_prediction.ipynb  ← Analysis notebook
├── housing.csv                   ← Dataset
├── requirements.txt              ← Dependencies
└── README.md                     ← This file
```

---

## 🚀 How to Run

### Option 1 — Live App (No installation needed!)
👉 [https://house-price-khushi.streamlit.app/](https://house-price-khushi.streamlit.app/)

### Option 2 — Run Locally

```bash
# Step 1 - Clone the repo
git clone https://github.com/Codewithkhushi-arch/house-price-prediction.git

# Step 2 - Go into the folder
cd house-price-prediction

# Step 3 - Install dependencies
pip install -r requirements.txt

# Step 4 - Run the app
streamlit run app.py
```

### Option 3 — View the Notebook

```bash
jupyter notebook house_price_prediction.ipynb
```

---

## 🔑 Key Learnings

- How to build end-to-end ML pipelines with Scikit-learn
- Feature engineering to improve model performance
- Handling missing values and categorical encoding
- Cross validation for reliable model evaluation
- Hyperparameter tuning with GridSearchCV
- Building and deploying interactive ML apps with Streamlit
- Version control with Git and GitHub

---

## 🔮 Future Improvements

- Add XGBoost model for comparison
- Include real-time data from housing APIs
- Add neighbourhood level crime and school data
- Implement SHAP values for model explainability
- Add user authentication for saving predictions

---

## 👩‍💻 Author

**Khushi** — Aspiring Data Scientist

- 🐙 GitHub: [@Codewithkhushi-arch](https://github.com/Codewithkhushi-arch)
- 🌐 Live App: [house-price-khushi.streamlit.app](https://house-price-khushi.streamlit.app/)

---

## 📚 References

- [California Housing Dataset — Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

⭐ If you found this project helpful, please give it a star on GitHub!
