import argparse
import logging
import os

import joblib
import pandas as pd

from utils import load_params

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "prediction.log"),
    level=logging.INFO,
    format="%(levelname)s: %(asctime)s %(message)s",
)

# Adding a StreamHandler to also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s %(message)s"))
logging.getLogger().addHandler(console_handler)

run_timestamp = pd.to_datetime("now").strftime("%Y%m%d_%H-%M-%S")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)

    DATA_PATH = params["predict"]["data_path"]
    MODEL_PATH = params["predict"]["model_path"]
    RESULTS_PATH = params["predict"]["results_path"]

    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Read unseen test data
    df_test = pd.read_csv(f"{DATA_PATH}/test_data.csv")
    df_test["obs"] = df_test["obs"].astype(int)
    # Load model and read features
    model = joblib.load(f"{MODEL_PATH}/rf_model.pkl")
    features = model.feature_names_in_

    prediction = model.predict(df_test[features])
    df_test["salary_category"] = prediction
    df_test["salary_category"] = df_test["salary_category"].map({0: "Low", 1: "Medium", 2: "High"})

    df_test = df_test[["obs", "salary_category"]].reset_index(drop=True)
    df_test.sort_values(by="obs", ascending=True, inplace=True)
    df_test = df_test.reset_index(drop=True)
    df_test.to_csv(f"{RESULTS_PATH}/{run_timestamp}_rf_model_results.csv", index=False)
