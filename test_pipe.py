import joblib

pipeline = joblib.load("data/pipeline.pkl")
print(pipeline.named_steps['preprocessor'].get_feature_names_out())