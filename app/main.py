from fastapi import FastAPI, HTTPException, Query, Request
import structlog
import logging
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from enum import Enum
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

POPULATION_DATASET = "zipcode_population.csv"
LOG_PATH = "app.log"
PIPELINE_PATH = "pipeline.pkl"

def preprocess_data(df, population_df):
    """
    Perform initial data preparation before the main pipeline.

    This function merges the input DataFrame with population data on 'zip_code',
    then drops columns that are not intended to be used as features in the model.

    Parameters:
        df (pandas.DataFrame): The main dataset containing customer information.
        population_df (pandas.DataFrame): Dataset containing population data keyed by 'zip_code'.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame ready for further processing or modeling.
    """
    # First merge with population data
    df = df.merge(population_df, on="zip_code", how="left")
    
    # Then drop columns that shouldn't be features
    df = df.drop(columns=[
        "customer_id", 
        "latitude", 
        "longitude", 
        "zip_code"
    ])
    
    return df

# Lifespan handler to replace deprecated startup event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager for application lifespan events"""
    # Initialize logging
    setup_logging()
    
    # Generate sample log entries
    logger = structlog.get_logger()
    logger.info("Application starting up")
    for i in range(3):
        logger.info("Sample log message", message_id=i+1)
    
    logger.info("Loading ML model")
    
    # Loading the preprocessor & the model
    pipeline = joblib.load(PIPELINE_PATH)
    
    app.state.pipeline = pipeline
    app.state.preprocessor = pipeline.named_steps['preprocessor']
    app.state.model = pipeline.named_steps['classifier']
    
    logger.info("Model loading completed")
    
    yield  # App runs here
    
    # Cleanup on shutdown
    logger.info("Application shutting down")

title = "Random Forest Prediction API"
summary = "FastAPI service for real-time predictions using a trained Random Forest classifier via a RESTful endpoint."
description = "FastAPI application for serving predictions using a trained Random Forest classifier. Accepts input features via REST endpoints, returns classification results, and integrates with MLflow for model tracking and version control."

app = FastAPI(
    title=title,
    description=description,
    summary=summary,
    version="1.0.0",
    lifespan=lifespan
)

def setup_logging():
    """Configure structlog with proper file handling"""
    
    # Create file if it doesn't exist
    if not os.path.exists(LOG_PATH):
        open(LOG_PATH, "a").close()
    
    # Configure structlog to use standard logging with FileHandler
    logging.basicConfig(
        filename=LOG_PATH,
        format="%(message)s",
        level=logging.INFO
    )
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False
    )

# Middleware to log all API requests and latency
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests and their latency"""
    # Get structlog logger
    logger = structlog.get_logger()
    
    # Prepare base log data
    log_data: Dict[str, Any] = {
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "client": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent")
    }
    
    # Record start time
    start_time = time.perf_counter()
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate latency
        latency = time.perf_counter() - start_time
        
        # Add response info to log
        log_data.update({
            "status_code": response.status_code,
            "latency": f"{latency:.4f} seconds"
        })
        
        # Log successful request
        logger.info("API Request", **log_data)
        return response
        
    except Exception as e:
        # Calculate latency even for exceptions
        latency = time.perf_counter() - start_time
        
        # Add exception info to log
        log_data.update({
            "status_code": 500,
            "latency": f"{latency:.4f} seconds",
            "exception": str(e)
        })
        
        # Log error
        logger.error("API Request Failed", **log_data)
        
        # Re-raise exception for proper error handling
        raise

@app.get("/api/health", 
         summary="Service Health Check",
         response_description="Service status information",
         tags=["Monitoring"])
async def health_check(request: Request):
    """
    Check the health status of the service.
    
    Returns a dictionary with:
    - **status**: Current service status ("healthy")
    - **timestamp**: UTC timestamp in ISO format
    - **model_status**: ML model status
    """
    model = request.app.state.model
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_status": "loaded" if model else "unavailable"
    }

@app.get("/api/logs",
         summary="Retrieve Application Logs",
         response_description="List of structured log entries",
         tags=["Logs"])
async def get_logs(
    count: Optional[int] = Query(
        default=10, 
        gt=0, 
        le=100, 
        title="Log Count",
        description="Number of recent log entries to retrieve (1-100)"
    )
) -> List[dict]:
    """
    Retrieve the most recent application logs in structured JSON format.
    
    Parameters:
    - **count**: Number of log entries to return (default: 10, max: 100)
    
    Returns:
    - List of JSON objects representing log entries, newest first
    - 404 error if log file doesn't exist
    - 500 error for processing failures
    """
    
    if not os.path.exists(LOG_PATH):
        raise HTTPException(
            status_code=404,
            detail="Log file not found"
        )
    
    try:
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
        
        # Process lines in reverse order (newest first)
        logs = []
        line_count = 0
        
        # Iterate from end of file backwards
        for line in reversed(lines):
            if line_count >= count:
                break
                
            stripped_line = line.strip()
            if not stripped_line:
                continue
                
            try:
                # Parse JSON safely
                log_entry = json.loads(stripped_line)
                logs.append(log_entry)
                line_count += 1
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
                
        return logs
    
    except Exception as e:
        # Log error using structlog
        logger = structlog.get_logger()
        logger.error("Log retrieval failed", error=str(e))
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing logs: {str(e)}"
        )
        
# List of Expected Columns
COLUMNS = [
    "Customer_ID",
    "Gender",
    "Age",
    "Married",
    "Number_of_Dependents",
    "City",
    "Zip_Code",
    "Latitude",
    "Longitude",
    "Number_of_Referrals",
    "Tenure_in_Months",
    "Offer",
    "Phone_Service",
    "Avg_Monthly_Long_Distance_Charges",
    "Multiple_Lines",
    "Internet_Service",
    "Internet_Type",
    "Avg_Monthly_GB_Download",
    "Online_Security",
    "Online_Backup",
    "Device_Protection_Plan",
    "Premium_Tech_Support",
    "Streaming_TV",
    "Streaming_Movies",
    "Streaming_Music",
    "Unlimited_Data",
    "Contract",
    "Paperless_Billing",
    "Payment_Method",
    "Monthly_Charge",
    "Total_Charges",
    "Total_Refunds",
    "Total_Extra_Data_Charges",
    "Total_Long_Distance_Charges",
    "Total_Revenue"
]


# Enums for categorical fields
class GenderEnum(str, Enum):
    Male = "Male"
    Female = "Female"

class YesNoEnum(str, Enum):
    Yes = "Yes"
    No = "No"

class InternetTypeEnum(str, Enum):
    DSL = "DSL"
    Fiber_Optic = "Fiber Optic"
    Cable = "Cable"
    None_ = "None"

class ContractEnum(str, Enum):
    Month_to_Month = "Month-to-Month"
    One_Year = "One Year"
    Two_Year = "Two Year"

class PaymentMethodEnum(str, Enum):
    Bank_Transfer = "Bank Transfer"
    Credit_Card = "Credit Card"
    Electronic_Check = "Electronic Check"
    Mailed_Check = "Mailed Check"

class PredictionInput(BaseModel):
    Customer_ID: str = Field(..., description="Unique customer identifier")
    Gender: GenderEnum
    Age: int = Field(..., gt=0, le=120)
    Married: YesNoEnum
    Number_of_Dependents: int = Field(..., ge=0, le=10)
    City: str
    Zip_Code: int = Field(..., gt=90000, le=100000)
    Latitude: float
    Longitude: float
    Number_of_Referrals: int = Field(..., ge=0)
    Tenure_in_Months: int = Field(..., ge=0)
    Offer: str
    Phone_Service: YesNoEnum
    Avg_Monthly_Long_Distance_Charges: float = Field(..., ge=0)
    Multiple_Lines: YesNoEnum
    Internet_Service: YesNoEnum
    Internet_Type: InternetTypeEnum
    Avg_Monthly_GB_Download: float = Field(..., ge=0)
    Online_Security: YesNoEnum
    Online_Backup: YesNoEnum
    Device_Protection_Plan: YesNoEnum
    Premium_Tech_Support: YesNoEnum
    Streaming_TV: YesNoEnum
    Streaming_Movies: YesNoEnum
    Streaming_Music: YesNoEnum
    Unlimited_Data: YesNoEnum
    Contract: ContractEnum
    Paperless_Billing: YesNoEnum
    Payment_Method: PaymentMethodEnum
    Monthly_Charge: float = Field(..., ge=0)
    Total_Charges: float = Field(..., ge=0)
    Total_Refunds: float = Field(..., ge=0)
    Total_Extra_Data_Charges: float = Field(..., ge=0)
    Total_Long_Distance_Charges: float = Field(..., ge=0)
    Total_Revenue: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "Customer_ID": "12345-ABCDE",
                "Gender": "Female",
                "Age": 45,
                "Married": "Yes",
                "Number_of_Dependents": 2,
                "City": "Los Angeles",
                "Zip_Code": "90001",
                "Latitude": 34.0522,
                "Longitude": -118.2437,
                "Number_of_Referrals": 3,
                "Tenure_in_Months": 24,
                "Offer": "Offer_E",
                "Phone_Service": "Yes",
                "Avg_Monthly_Long_Distance_Charges": 12.75,
                "Multiple_Lines": "Yes",
                "Internet_Service": "Yes",
                "Internet_Type": "Fiber Optic",
                "Avg_Monthly_GB_Download": 125.5,
                "Online_Security": "Yes",
                "Online_Backup": "No",
                "Device_Protection_Plan": "Yes",
                "Premium_Tech_Support": "No",
                "Streaming_TV": "Yes",
                "Streaming_Movies": "Yes",
                "Streaming_Music": "No",
                "Unlimited_Data": "Yes",
                "Contract": "Two Year",
                "Paperless_Billing": "Yes",
                "Payment_Method": "Credit Card",
                "Monthly_Charge": 89.99,
                "Total_Charges": 2159.76,
                "Total_Refunds": 0.0,
                "Total_Extra_Data_Charges": 15.0,
                "Total_Long_Distance_Charges": 153.0,
                "Total_Revenue": 2312.76
            }
        }
        
class PredictionOutput(BaseModel):
    prediction: int
    probabilities: List[float]
    class_labels: List[int]

@app.post("/api/predictions",
          summary="Make Customers Churn Prediction using Random Forest Classifier",
          response_description="Prediction results",
          tags=["random_forest"])
async def predict(
    inputs: List[PredictionInput],
    request: Request
) -> List[PredictionOutput]:
    """
    Make predictions using the trained Random Forest model.
    
    Parameters:
    - **inputs**: List of input records with features
    
    Returns:
    - List of prediction objects containing:
        - prediction: Predicted class
        - probabilities: Class probabilities
        - class_labels: Available class labels
    """
    logger = structlog.get_logger()
    logger.info("Prediction request received", 
                client=request.client.host if request.client else "unknown",
                num_records=len(inputs))
    # Load preprocessor & model
    preprocessor = request.app.state.preprocessor
    model = request.app.state.model
    
    try:
        # Convert input to DataFrame (preserving original case)
        input_data = [item.model_dump() for item in inputs]
        df = pd.DataFrame(input_data)
        
        # Renaming the column names
        df = df.rename(
            lambda x: x.lower().strip().replace(' ', '_'), 
            axis='columns'
        )
        
        # Generating list of the expected columns
        expected_columns = [s.lower().replace(" ", "_") for s in COLUMNS]
        
        # Create case-insensitive mapping
        column_mapping = {col.lower(): col for col in df.columns}
        
        # Check for missing columns
        missing_cols = set(expected_columns) - set(column_mapping.keys())
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_cols}"
            )
        
        # Loads the population Dataset
        population_df = pd.read_csv(POPULATION_DATASET)
        
        # Converting all column names to lower letters and replacing spaces with underscores.
        population_df = population_df.rename(
            lambda x: x.lower().strip().replace(' ', '_'),
            axis="columns"
        )

        # Preprocess data with population merge
        df = preprocess_data(df, population_df)
        
        print(df.columns)

        # Preprocess features
        logger.info("Preprocessing features")
        
        processed_features = preprocessor.transform(df)
        
        # Make predictions
        logger.info("Making predictions")
        probabilities = model.predict_proba(processed_features)
        predictions = np.argmax(probabilities, axis=1)
        
        # Format response
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "prediction": int(pred),
                "probabilities": probabilities[i].tolist(),
                "class_labels": model.classes_.tolist()
            })
            
        logger.info("Prediction completed", 
                    num_predictions=len(results),
                    prediction_sample=results[0] if results else None)
        
        return results
    
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
