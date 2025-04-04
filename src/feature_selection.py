import argparse
import numpy as np
import logging
import pandas as pd 
import os
from sklearn.feature_selection import chi2
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from utils import load_params

""" 
Steps:
1. Read train dataset.
2. Drop "obs" column.
3. Run Feature selection using Chi-Squared test.
4. Record the selected features.
"""

log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'feature_selection.log'),
                    level=logging.INFO,
                    format='%(levelname)s: %(asctime)s %(message)s') #Can optionally add %(processName)s 

# Adding a StreamHandler to also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s %(message)s'))
logging.getLogger().addHandler(console_handler)

#run_timestamp = pd.to_datetime('now').strftime('%Y%m%d_%H-%M-%S')

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)

    DATA_INPUT_PATH = params["feature_selection"]["data_input_path"]
    DATA_OUTPUT_PATH = params["feature_selection"]["data_output_path"]
    N_FEATURES = params["feature_selection"]["number"]

    os.makedirs(DATA_OUTPUT_PATH, exist_ok=True)

    df = pd.read_csv(f"{DATA_INPUT_PATH}/training_data.csv")   #"../data/processed/training_data.csv")
    df_valid = pd.read_csv(f"{DATA_INPUT_PATH}/validation_data.csv")  #("../data/processed/validation_data.csv")
    # Drop "obs" column
    # Separate features and target variable
    X = df.drop(columns=["obs", "salary_category_num", "salary_category"])
    y = df["salary_category_num"]
    
    selector = SelectKBest(score_func=chi2, k=N_FEATURES)
    feature_names = X.columns
    #fit = selector.fit(X_fs, y)
    X_new = selector.fit_transform(X, y)

    # Get selected feature names
    selected_features = feature_names[selector.get_support()]
    logging.info(f"{N_FEATURES} Selected features: {selected_features}")
    df_output = df[selected_features]
    df_output['salary_category'] = df['salary_category']
    df_output.to_csv(f"{DATA_OUTPUT_PATH}/training_data_fs.csv", index=False) #f'../data/processed/training_data_fs.csv', index=False)

    df_valid_output = df_valid[selected_features]
    df_valid_output['salary_category'] = df_valid['salary_category']
    df_valid_output.to_csv(f"{DATA_OUTPUT_PATH}/validation_data_fs.csv", index=False) #(f'../data/processed/validation_data_fs.csv', index=False)
