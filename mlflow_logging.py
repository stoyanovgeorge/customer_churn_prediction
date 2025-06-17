# Needed to Parse the CLI Arguments
import os
import sys
import argparse
from pathlib import Path
from urllib.parse import urlparse

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

def parse_max_depth(value):
    """Custom type function for argument parser to handle max_depth.
    
    Converts:
    - 0 → None (unlimited depth)
    - Positive integers → integer
    - None/"None" → None
    """
    if value is None or str(value).strip().lower() == "none":
        return None
    try:
        depth = int(value)
        if depth < 0:
            raise argparse.ArgumentTypeError("max_depth must be non-negative")
        return None if depth == 0 else depth
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid max_depth: '{value}'. Must be integer >=0 or None"
        )

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
        mlflow_uri="http://localhost:5000"
    ):
    """
    Cleans the input dataset, trains a Random Forest classifier using the specified 
    hyperparameters, and logs the model, parameters, and artifacts to an MLflow server.

    The function performs the following steps:
    - Loads and preprocesses the dataset.
    - Trains a Random Forest classifier within a pipeline.
    - Logs hyperparameters, metrics, and the trained pipeline to the specified MLflow tracking URI.

    Parameters:
        dataset (str or Path): Path to the input CSV dataset.
        n_estimators (int): Number of trees in the Random Forest.
        max_depth (int, optional): Maximum depth of each tree. Defaults to None.
        min_ssplit (int): Minimum number of samples required to split an internal node.
        max_feats (str or int or float): Number of features to consider when looking for the best split.
        mlflow_uri (str): URI of the MLflow tracking server. Defaults to "http://localhost:5000".

    Returns:
        Trained pipeline (sklearn.pipeline.Pipeline): The full pipeline including preprocessing and the classifier.
    """
    # Move imports inside the function to speed up script startup
    import mlflow
    import pickle
    import warnings
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        OneHotEncoder,
        LabelEncoder,
        OrdinalEncoder
    )
    from sklearn.metrics import (
        classification_report, 
        recall_score, 
        precision_score, 
        f1_score, 
        roc_auc_score,
        accuracy_score,
        ConfusionMatrixDisplay
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

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
        ('encoder', OrdinalEncoder())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="Missing")),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ])
        
    from category_encoders import CountEncoder
    preprocessor = ColumnTransformer(
    transformers=[
        ('binary', binary_transformer, binary_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols),
        ('count_enc', CountEncoder(normalize=False), ['city']) 
    ],
    remainder='drop',
    verbose_feature_names_out=False
    )

    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            min_samples_split=min_ssplit,
            n_jobs=-1
        ))
    ])
    
    # Prepare data
    X = df.drop(columns=["customer_status"])
    le = LabelEncoder()
    y = le.fit_transform(df["customer_status"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Suppressing the warning about missing values in MLflow
    # Reference: https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Integer columns in Python cannot represent missing values.*")

        
    # MLflow setup
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("RF Churn Prediction")
    
    dataset_name = get_dataset_name(df)
    run_name = f"{dataset_name} | NE={n_estimators} | MD={max_depth} | MSS={min_ssplit}"

    with mlflow.start_run(run_name=run_name):
        mlflow.sklearn.autolog()
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        test_accuracy = accuracy_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred, average="weighted")  # Use "binary" for binary
        test_precision = precision_score(y_test, y_pred, average="weighted")
        test_f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_f1", test_f1)
        
        # Push the whole pipeline to MLflow
        pipeline_path = "app/pipeline.pkl"
        
        model_data = {
            "pipeline": pipeline,
            "label_encoder": le
        }
        
        with open(pipeline_path, "wb") as f:
            pickle.dump(model_data, f)
            
        mlflow.log_artifact(pipeline_path)
        
        # Removing the pipeline from the data/ directory
        os.remove(pipeline_path)
        
        # Manually logs the AUC Macro (Generic, works for both binary & multiclass classification)         
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        mlflow.log_metric("auc", auc)
        
        model = pipeline.named_steps['classifier']
        importances = model.feature_importances_
        feature_names = pipeline[:-1].get_feature_names_out()
        
        # Create DataFrame of the Features
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

        # Clean up and Beautify the feature names
        feat_df['Feature'] = (feat_df['Feature']
                            .str.replace('remainder__', '')  # Remove remainder prefix
                            .str.replace('_', ' ')  # Replace underscores with spaces
                            .str.replace("offer", "", 1) # Replace the first occurence of the word
                            .str.strip()  # Remove any leading/trailing whitespace
                            .str.title()  # Capitalize first letter of each word
                            )

        # Sort the features in descending order
        feat_df = feat_df.sort_values(by='Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(11,6))
        sns.barplot(data=feat_df, x='Importance', y='Feature')
        plt.title("Feature Importances (Sorted)")
        plt.tight_layout()
        plt.savefig("data/features_imporance.png")
        mlflow.log_artifact("data/features_imporance.png")       
        
        # Confusion matrix
        # ConfusionMatrixDisplay.from_estimator(
        #     pipeline,
        #     X_test,
        #     y_test,
        #     display_labels=["Churned", "Joined", "Stayed"],
        #     cmap='Blues'
        # )
        # plt.title("Confusion Matrix")
        # plt.tight_layout()
        # plt.savefig("data/testing_confusion_matrix_labeled.png")
        # mlflow.log_artifact("data/testing_confusion_matrix_labeled.png")
        # plt.close()
        
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
        "--max_depth", type=parse_max_depth, default=None, metavar="DEPTH",
        help="Maximum depth of the trees"
    )
    parser.add_argument(
        "--min_ssplit", type=int, default=2, metavar="N",
        help="Defines the minimum number of samples required to split an internal node in a Random Forest."
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
        min_ssplit=args.min_ssplit,
        mlflow_uri=args.mlflow_uri
    )