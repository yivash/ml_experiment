import argparse
import numpy as np
import logging
import os
import pandas as pd 
from sklearn.feature_selection import chi2
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils import load_params

""" 
Steps:
1. Read train and test data.
2. Drop timestamp column ("job_posted_date").
3. Impute missing values (using K-Means).
4. Target encode categorical features.
5. Scale numerical features.
6. Transform Boolean columns to binary (0, 1).
7. Split train data into train and validation sets.
8. Save all datasets (train, validation, test) to "processed" folder.
"""

log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'data_preparation.log'),
                    level=logging.INFO,
                    format='%(levelname)s: %(asctime)s %(message)s') #Can optionally add %(processName)s 

# Adding a StreamHandler to also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s %(message)s'))
logging.getLogger().addHandler(console_handler)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)

    INPUT_DATA_PATH = params["data_prep"]["input_data_path"]
    OUTPUT_DATA_PATH = params["data_prep"]["output_data_path"]

    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Read train and test data
    train_data = pd.read_csv(f"{INPUT_DATA_PATH}/train.csv")
    test_data = pd.read_csv(f"{INPUT_DATA_PATH}/test.csv")

    # Drop timestamp column ("job_posted_date")
    train_data.drop(columns=["job_posted_date"], inplace=True)
    test_data.drop(columns=["job_posted_date"], inplace=True)

    # Impute missing values
    train_data['job_state'] = train_data['job_state'].fillna(train_data['job_state'].mode()[0])
    test_data['job_state'] = test_data['job_state'].fillna(test_data['job_state'].mode()[0])

    imputer = KNNImputer()
    num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
    train_data[num_cols] = imputer.fit_transform(train_data[num_cols])
    test_data[num_cols] = imputer.transform(test_data[num_cols])

    # Target encode categorical features
    train_data['salary_category_num'] = train_data['salary_category'].map({'High': 2, 'Medium': 1, 'Low':0})
    categorical_features = ["job_title", "job_state", "feature_1"]
    for feature in categorical_features:
        df_stats = train_data.groupby(feature).agg({'salary_category_num': 'mean'}).reset_index()
        mapping = df_stats.set_index(feature)['salary_category_num'].to_dict()
        train_data[feature] = train_data[feature].map(mapping)
        test_data[feature] = test_data[feature].map(mapping)
    
    # Transform boolean features
    for col in [col for col in train_data.columns if train_data[col].dtype == 'bool']:
        train_data[col] = train_data[col].map({True: 1, False: 0})
        test_data[col] = test_data[col].map({True: 1, False: 0})

    # Scale numerical features
    features_to_scale = [col for col in train_data.columns if 'job_desc' in col]
    scaler = MinMaxScaler()
    train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
    test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])

    # Split train data into train and validation sets
    df_train, df_validation = train_test_split(
        train_data,
        test_size=0.2,
        random_state=42,
        stratify=train_data["salary_category"]
    )

    # Save all datasets (train, validation, test) to "processed" folder
    os.makedirs(log_dir, exist_ok=True)
    df_train.to_csv(f"{OUTPUT_DATA_PATH}/training_data.csv", index=False)
    df_validation.to_csv(f"{OUTPUT_DATA_PATH}/validation_data.csv", index=False)
    test_data.to_csv(f"{OUTPUT_DATA_PATH}/test_data.csv", index=False)

    logging.info("Data preparation completed and saved to 'processed' folder.")