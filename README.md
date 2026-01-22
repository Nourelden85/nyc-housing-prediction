# New York Housing Market: End-to-End Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)

## Project Overview

An end-to-end Machine Learning solution designed to estimate real estate prices in New York City. The project moves from raw data scraping/collection to a fully functional interactive web application, helping users understand market trends and property values.

## Key Performance Indicators (KPIs)

- **Model Accuracy ($R^2$):** 86% on unseen data.
- **Error Metric (MAPE):** 28% Mean Absolute Percentage Error.
- **Model Validation on Real-World Data (Zillow Test Cases):**

| Locality      | Property Type | Zip Code | Coordinates (Lat, Long) | Sqft  | Actual Price (Zillow) | Model Prediction | Accuracy |
| :------------ | :------------ | :------- | :---------------------- | :---- | :-------------------- | :--------------- | :------- |
| **Manhattan** | Condo         | 10021    | 40.7661, -73.9641       | 2,200 | \$3,850,000           | \$3,371,963      | 87.5%    |
| **Brooklyn**  | Townhouse     | 11211    | 40.7105, -73.9625       | 1,150 | \$1,250,000           | \$1,271,603      | 98.3%    |
| **Queens**    | Multi-family  | 11103    | 40.7668, -73.9161       | 950   | \$875,000             | \$600,413        | 68.6%    |

## Technical Highlights

- **Feature Engineering:** Implemented **Haversine Distance** algorithms to calculate proximity to major NYC landmarks (Times Square & Wall Street), capturing the "Location, Location, Location" factor.
- **Data Pipeline:** Built a robust `ColumnTransformer` pipeline for automated encoding of categorical variables and scaling of numerical features.
- **Advanced Modeling:** Leveraged `Random Forest Regressor` with **Log Transformation** on the target variable to handle price skewness and improve convergence.
- **Interactive UI:** Developed a `Streamlit` dashboard featuring an integrated `Folium` map for real-time geographic coordinate selection.

## Project Architecture

```text
├── app.py                  # Streamlit Web Application
├── data/
│   ├── raw/                # Original dataset
│   ├── cleaned/            # Cleaned dataset
│   └── processed/          # Featured engineered data ready for modeling
├── models/
│   ├── model.joblib        # Trained Random Forest Model
│   └── preprocessor.joblib # Fitted Scikit-Learn Pipeline
├── notebooks/
│   ├── 01_data_loading_and_overview.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_preprocessing
│   └── 05_modeling
├── requirements.txt        # Reproducibility list
└── README.md
```

## Installation & Usage

1. Clone the repo
   git clone [https://github.com/Nourelden85/nyc-housing-prediction.git](https://github.com/Nourelden85/nyc-housing-prediction.git)

2. Install dependencies
   pip install -r requirements.txt

3. Run the App
   streamlit run app.py
