import os
import pickle
import numpy as np
import pandas as pd
import kagglehub as kh
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from kagglehub import KaggleDatasetAdapter

# Importing the Label Encoder
from sklearn.preprocessing import LabelEncoder

title = "Random Forest Prediction API"
summary = "FastAPI service for real-time predictions using a trained Random Forest classifier via a RESTful endpoint."
description = "FastAPI application for serving predictions using a trained Random Forest classifier. Accepts input features via REST endpoints, returns classification results, and integrates with MLflow for model tracking and version control."

app = FastAPI(
    title=title,
    description=description,
    summary=summary,
    version="0.0.1",
    contact={
        "name": "Georgi Stoyanov"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

MODEL_PATH = "app/model.pkl"

# Check if model file exists and is non-empty
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    raise RuntimeError(f"Model file '{MODEL_PATH}' not found or is empty.")

# Load model from app/
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class PredictRequest(BaseModel):
    features: list[list[float]]  # List of feature vectors


def replace_nans(df, col_list, value):
    """
    Replace missing values in specified columns of a DataFrame with a given value.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        col_list (list of str): List of column names to fill NaNs in.
        value: The value to replace NaNs with.

    Returns:
        pd.DataFrame: DataFrame with NaNs replaced in the specified columns.
    """
    for col in col_list:
        df[col] = df[col].fillna(value)
    return df

def df_processing(df):
    # Retrieve the Population Dataset
    population_df = kh.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "shilongzhuang/telecom-customer-churn-by-maven-analytics",
        "telecom_zipcode_population.csv"
    )
    
    # Converting all column names to lower letters and replacing spaces with underscores.
    df = df.rename(
        lambda x: x.lower().strip().replace(' ', '_'), 
        axis='columns'
    )

    # Converting all column names to lower letters and replacing spaces with underscores.
    population_df = population_df.rename(
        lambda x: x.lower().strip().replace(' ', '_'),
        axis="columns"
    )
    
    # Merge df with population_df on 'zip_code'
    df = df.merge(population_df, on="zip_code", how="left")
    
    # Applying frequency encoding for the city features
    city_freq = df["city"].value_counts()
    df["city_encoded"] = df["city"].map(city_freq)
    
    # Dropping some columns
    df = df.drop(columns = [
        "customer_id",
        "latitude",
        "longitude",
        "churn_category",
        "churn_reason",
        "zip_code",
        "city"
    ])
    
    categorical_cols_with_nans = [
        "internet_type",
        "offer"
    ]

    df = replace_nans(df, categorical_cols_with_nans, "Missing")
    
    binary_cols_with_nans = [
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection_plan",
        "premium_tech_support",
        "streaming_tv",
        "streaming_movies",
        "streaming_music",
        "unlimited_data"
    ]
    
    df = replace_nans(df, binary_cols_with_nans, "No")
    
    numerical_cols_with_nans = [
        "avg_monthly_long_distance_charges",
        "avg_monthly_gb_download"
    ]

    df = replace_nans(df, numerical_cols_with_nans, 0)
    
    binary_string_cols = []

    for col in df.columns:
        unique_vals = df[col].unique()
        if len(unique_vals) == 2 and all(isinstance(val, str) for val in unique_vals):
            binary_string_cols.append(col)

    # Encode each column with a LabelEncoder instance
    for col in binary_string_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    
    # NEED EXPORTED ONEHOT ENCODER
    
    # categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    


@app.post("/api/v1/random-forest/classification")
async def predict(request: PredictRequest):
    try:
        input_array = np.array(request.features)
        predictions = model.predict(input_array)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
