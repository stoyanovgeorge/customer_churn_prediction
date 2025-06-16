# Needed to Parse the CLI Arguments
import os
import sys
import argparse
from pathlib import Path
from urllib.parse import urlparse

# Replaces NaN values with user-defined defaults
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

def get_dataset_name(dataset, small_threshold=1000):
    """
    Determine the dataset size category based on the number of rows.

    Args:
        dataset (pd.DataFrame): The dataset whose size will be evaluated.
        small_threshold (int, optional): The maximum number of rows for a dataset
            to be considered 'Small'. Defaults to 1000.

    Returns:
        str: "Small Dataset" if the dataset has rows less than or equal to the threshold,
            otherwise "Full Dataset".
    """
    return "Small DF" if dataset.shape[0] <= small_threshold else "Full DF"


def frequency_encode_city(df):
    """
    Perform frequency encoding on the 'city' column of a DataFrame.

    This function calculates the frequency of each unique city in the 'city' column,
    then creates a new column 'city_encoded' where each city is replaced by its frequency count.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing a 'city' column.

    Returns:
        pandas.DataFrame: A new DataFrame with an added 'city_encoded' column representing
                          the frequency encoding of the 'city' column.
    """
    city_freq = df["city"].value_counts()
    return df.assign(city_encoded=df["city"].map(city_freq))

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
    # Merge with population data
    df = df.merge(population_df, on="zip_code", how="left")
    
    # Drop columns that shouldn't be features
    df = df.drop(columns=[
        "customer_id", 
        "latitude", 
        "longitude", 
        "churn_category", 
        "churn_reason", 
        "zip_code"
    ])
    
    return df

def validate_mlflow_uri(uri):
    """Validates and normalizes an MLflow tracking URI.
    
    Performs basic validation of the URI structure according to MLflow's expected
    tracking URI formats. Supports HTTP, HTTPS, file, and SQLite URI schemes.

    Args:
        uri: The tracking URI to validate. Expected formats:
            - HTTP/HTTPS: 'http://host:port' or 'https://host:port'
            - File: 'file:///absolute/path' 
            - SQLite: 'sqlite:///relative/path.db'

    Returns:
        Normalized URI with trailing slashes removed.

    Raises:
        SystemExit: If the URI is invalid, with an explanatory error message
        that includes valid format examples.

    Examples:
        >>> validate_mlflow_uri("http://localhost:5000")
        'http://localhost:5000'

        >>> validate_mlflow_uri("file:///mlruns")
        'file:///mlruns'
    """
    if not uri:
        sys.exit("Error: Empty MLflow tracking URI")
    
    parsed = urlparse(uri)
    if not parsed.scheme:
        sys.exit(f"Error: Missing scheme in URI '{uri}'\n"
                "Valid formats: http://host:port, file://path, sqlite://path")
    
    if parsed.scheme in ('http', 'https') and not parsed.netloc:
        sys.exit(f"Error: Missing host in URI '{uri}'")
    
    return uri.rstrip('/')


def train_random_forest_classifier(
        dataset, 
        n_estimators, 
        max_depth=None, 
        min_ssplit=2,
        max_feats="sqrt",
        mlflow_uri="http://localhost:5000"
    ):
    # Move imports inside the function to speed up script startup
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        OneHotEncoder,
        FunctionTransformer,
        LabelEncoder
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        classification_report,
        ConfusionMatrixDisplay
    )
    import mlflow
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns

    # from kagglehub import KaggleDatasetAdapter

    # Importing the Label Encoder
    from sklearn.preprocessing import LabelEncoder

    # Importing Plotting Libraries
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the Customer's Churn & Population Dataset
    df = pd.read_csv(dataset)
    population_path = "data/zipcode_population.csv"
    population_df = pd.read_csv(population_path)

    # Renaming the columns in the main DF, containing the customers data
    df = df.rename(
        lambda x: x.lower().strip().replace(' ', '_'), 
        axis='columns'
    )

    # Converting all column names to lower letters and replacing spaces with underscores.
    population_df = population_df.rename(
        lambda x: x.lower().strip().replace(' ', '_'),
        axis="columns"
    )

    # Dropping unneeded columns 
    df = preprocess_data(df, population_df)

    binary_cols = [
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
    
    categorical_cols = [
        "internet_type", 
        "offer"
    ]
    
    numerical_cols = [
        "avg_monthly_long_distance_charges", 
        "avg_monthly_gb_download",
        "population"
    ]

    # Create preprocessing pipelines
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="No")),
        ('encoder', FunctionTransformer(
            lambda x: LabelEncoder().fit_transform(x.astype(str))
        ))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="Missing")),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        # Removed StandardScaler since Random Forest doesn't need it
    ])
    
    city_transformer = Pipeline(steps=[
        ('encoder', FunctionTransformer(frequency_encode_city)),
        ('selector', FunctionTransformer(lambda x: x[["city_encoded"]]))
    ])
    
    # ColumnTransformer for all features
    preprocessor = ColumnTransformer(
        transformers=[
            ('binary', binary_transformer, binary_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols),
            ('city', city_transformer, ['city'])
        ],
        remainder='drop'
    )
    
    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            max_features=max_feats,
            min_samples_split=min_ssplit,
            n_jobs=-1
        ))
    ])
    
    # Prepare data
    X = df.drop(columns=["customer_status"])
    y = LabelEncoder().fit_transform(df["customer_status"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # MLflow setup
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("RF Churn Prediction")
    
    dataset_name = get_dataset_name(df)
    run_name = f"{dataset_name} | NE={n_estimators}, MD={max_depth}"

    with mlflow.start_run(run_name=run_name):
        mlflow.sklearn.autolog()
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Save model
        with open("data/model.pkl", "wb") as f:
            pickle.dump(pipeline, f)
        mlflow.log_artifact("data/model.pkl")
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        # Manually logs the AUC Macro (Generic, works for both binary & multiclass classification)         
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        mlflow.log_metric("auc", auc)
        
        # Feature importance plot
        feature_names = (
            binary_cols + 
            list(pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(categorical_cols)) +
            numerical_cols +
            ['city_encoded']
        )
        
        importance = pd.Series(
            pipeline.named_steps['classifier'].feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=importance.index)
        plt.title("Feature Importance (Sorted)")
        plt.tight_layout()
        plt.savefig("data/feature_importance.png")
        mlflow.log_artifact("data/feature_importance.png")
        plt.close()
        
        # Confusion matrix
        ConfusionMatrixDisplay.from_estimator(
            pipeline,
            X_test,
            y_test,
            display_labels=["Churned", "Joined", "Stayed"],
            cmap='Blues'
        )
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("data/confusion_matrix.png")
        mlflow.log_artifact("data/confusion_matrix.png")
        plt.close()
        
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest classifier and log metrics/artifacts to an MLflow server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-D", "--dataset", type=str, required=True, metavar="FILE",
        help="Path to the CSV dataset"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=100, metavar="N",
        help="Number of trees in the forest"
    )
    parser.add_argument(
        "--max_depth", type=int, default=None, metavar="DEPTH",
        help="Maximum depth of the trees"
    )
    parser.add_argument(
        "--min_ssplit", type=int, default=2, metavar="N",
        help="Defines the minimum number of samples required to split an internal node in a Random Forest."
    )
    parser.add_argument(
        "--max_feats", type=str, default="sqrt", metavar="VALUE",
        help="specifies the number of features to consider when looking for the best split at each node in a Random Forest."
    )
    parser.add_argument(
        "--mlflow_uri", type=str, default="http://localhost:5000", metavar="URI",
        help="Specifies the tracking URI where MLflow logs will be recorded"
    )    

    args = parser.parse_args()

    # Checking if the dataset path exists
    if not os.path.isfile(args.dataset):
        print(f"Error: Dataset file does not exist at {args.dataset}")
        sys.exit(1)
    
    # Checking if the provided file is a valid CSV file
    if not args.dataset.endswith(".csv"):
        print(f"Error: Dataset file is not a valid CSV file: {args.dataset}")
        sys.exit(1)
    
    dataset_path = Path(args.dataset)
    # Check if the dataset file is not empty
    # if args.dataset.stat().st_size == 0:
    if dataset_path.stat().st_size == 0:    
        print(f"Error: File is empty: {args.dataset}")
        sys.exit(1)

    # Check if the provided mflow_uri is valid
    mlflow_uri = args.mlflow_uri
    validate_mlflow_uri(mlflow_uri)
    
    # Lower the case of the user input
    max_features = args.max_feats.lower()
    if max_features not in ("sqrt", "log2"):
        raise ValueError(
            f"Invalid max_features value: '{max_features}'. "
            "Allowed values are 'sqrt' or 'log2' (case insensitive)."
        )

    result = train_random_forest_classifier(
        dataset=args.dataset,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        mlflow_uri=args.mlflow_uri
    )