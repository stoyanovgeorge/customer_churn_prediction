# Needed to Parse the CLI Arguments
import os
import sys
import argparse
from urllib.parse import urlparse


def train_random_forest_classifier(
        dataset, 
        n_estimators, 
        max_depth=None, 
        mlflow_uri="http://localhost:5000"
    ):
    # Move imports inside the function to speed up script startup
    import os
    import pandas as pd
    import kagglehub as kh

    import mlflow

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    from kagglehub import KaggleDatasetAdapter

    # Importing the Label Encoder
    from sklearn.preprocessing import LabelEncoder

    # Importing Plotting Libraries
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(dataset)
    # population_df = pd.read_csv(population_dataset)

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

    # Merge df with population_df on 'zip_code'
    df = df.merge(population_df, on="zip_code", how="left")

    # Get current columns list
    cols = df.columns.tolist()

    # Remove 'city_encoded' from current position
    cols.remove("population")

    # Find index of 'city' column
    insertion_index = cols.index("city") + 1

    # Insert 'city_encoded' after 'city'
    cols.insert(insertion_index, "population")

    # Reorder DataFrame columns
    df = df[cols]

    # Removing features with low predictive value for Random Forest classification.
    df = df.drop(columns = [
        "customer_id",
        "latitude",
        "longitude",
        "churn_category",
        "churn_reason",
        "zip_code"
    ])

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

    # Replace NaN values in categorical features with the placeholder "Missing"
    categorical_cols_with_nans = [
        "internet_type",
        "offer"
    ]

    df = replace_nans(df, categorical_cols_with_nans, "Missing")

    # Replace NaN values in binary string features with the placeholder "No"
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

    # Replace NaN values in numerical features with the integer 0
    numerical_cols_with_nans = [
        "avg_monthly_long_distance_charges",
        "avg_monthly_gb_download"
    ]

    df = replace_nans(df, numerical_cols_with_nans, 0)

    # Fetching a list of columns with binary string features
    binary_string_cols = []

    for col in df.columns:
        unique_vals = df[col].unique()
        if len(unique_vals) == 2 and all(isinstance(val, str) for val in unique_vals):
            binary_string_cols.append(col)

    # Encode each column with a LabelEncoder instance
    for col in binary_string_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # Apply LabelEncoder to convert target class labels into integer values
    le = LabelEncoder()

    df["customer_status"] = le.fit_transform(df["customer_status"])

    # Apply frequency encoding to the values in the `city` column
    city_freq = df["city"].value_counts()
    df["city_encoded"] = df["city"].map(city_freq)

    # Re-order the `city_encoded` column and drop the original `city` column
    # Get current columns list
    cols = df.columns.tolist()

    # Remove 'city_encoded' from current position
    cols.remove("city_encoded")

    # Find index of 'city' column
    insertion_index = cols.index("city") + 1

    # Insert 'city_encoded' after 'city'
    cols.insert(insertion_index, "city_encoded")

    # Reorder DataFrame columns
    df = df[cols]

    # Drop the original 'city' column
    df = df.drop(columns=["city"])

    # Apply OneHotEncoder on the categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

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
        return "Small Dataset" if dataset.shape[0] <= small_threshold else "Full Dataset"
    
    dataset_name = get_dataset_name(dataset)

    # Convert all int columns to float64 to suppress MLflow Warning
    for col in dataset.columns:
        if pd.api.types.is_integer_dtype(dataset[col]):
            dataset[col] = dataset[col].astype("float64")

    # Features and target
    X = dataset.drop(columns=["customer_status"])
    y = dataset["customer_status"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

        # Set up MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "RF Churn Prediction"
    mlflow.set_experiment(experiment_name)

    run_name = (
        f"{dataset_name} | Estimators={n_estimators}, Depth={max_depth}"
    )

    with mlflow.start_run(run_name=run_name):
        # Enable autologging before training
        mlflow.autolog()

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log AUC (handle binary or multiclass)
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        mlflow.log_metric("auc", auc)

        # Log classification report metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)
            elif isinstance(metrics, float):  # "accuracy"
                mlflow.log_metric(label, metrics)

        # Feature importance plot
        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance = importance.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=importance.index)
        plt.title("Feature Importance (Sorted)")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        # Confusion matrix plot
        label_map = [
            "Churned",
            "Joined",
            "Stayed"
        ]

        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=label_map,
            cmap='Blues'
        )
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("training_confusion_matrix.png")
        mlflow.log_artifact("training_confusion_matrix.png")
        plt.close()

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
        "--n-estimators", type=int, default=100, metavar="N",
        help="Number of trees in the forest"
    )
    parser.add_argument(
        "--max-depth", type=int, default=None, metavar="DEPTH",
        help="Maximum depth of the trees"
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
    
    # Check if the dataset file is not empty
    if args.dataset.stat().st_size == 0:
        print(f"Error: File is empty: {args.dataset}")
        sys.exit(1)

    # Check if the provided mflow_uri is valid
    parsed_uri = urlparse(args.mlflow_uri)
    if not ((parsed_uri.scheme in ("http", "https") and parsed_uri.netloc) or 
            (parsed_uri.scheme == "file" and os.path.exists(parsed_uri.path))):
        print(f"Error: Invalid MLflow tracking URI: '{args.mlflow_uri}'.")
        sys.exit(1)

    train_random_forest_classifier(
        dataset=args.dataset,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        mlflow_uri=args.mlflow_uri
    )