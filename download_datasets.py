import os
import kagglehub as kh
from kagglehub import KaggleDatasetAdapter

df = kh.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "shilongzhuang/telecom-customer-churn-by-maven-analytics",
    "telecom_customer_churn.csv"
)

df_small = df.sample(n=1000, random_state=42)

df_full = df.copy()

csv_path = "data"

if not os.path.exists(csv_path):
    os.makedirs(csv_path)

df_small.to_csv(f"{csv_path}/data_sample_1000.csv", index=False)
df_full.to_csv(f"{csv_path}/data_full.csv", index=False)