import argparse
from dvclive import Live
import joblib
import json
import os
import logging
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from utils import load_params

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "train_evaluate.log"),
    level=logging.INFO,
    format="%(levelname)s: %(asctime)s %(message)s",
)  # Can optionally add %(processName)s

# Adding a StreamHandler to also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s %(message)s"))
logging.getLogger().addHandler(console_handler)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)

    DATA_PATH = params["train_evaluate"]["data_path"]
    MODEL_PATH = params["train_evaluate"]["model_path"]
    MODEL_NAME = params["train_evaluate"]["model_name"]
    CV_RESULTS_PATH = params["train_evaluate"]["cv_results_path"]

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(CV_RESULTS_PATH, exist_ok=True)

    # Read train and validation data
    train_data = pd.read_csv(f"{DATA_PATH}/training_data_fs.csv")
    train_data["salary_category_num"] = train_data["salary_category"].map({"High": 2, "Medium": 1, "Low": 0})
    valid_data = pd.read_csv(f"{DATA_PATH}/validation_data_fs.csv")
    valid_data["salary_category_num"] = valid_data["salary_category"].map({"High": 2, "Medium": 1, "Low": 0})

    # Separate features and target variable
    X_train = train_data.drop(columns=["salary_category_num", "salary_category"])
    y_train = train_data["salary_category_num"]

    X_valid = valid_data.drop(columns=["salary_category_num", "salary_category"])
    y_valid = valid_data["salary_category_num"]
    
    if MODEL_NAME == "random_forest":
        # Initialize the model
        model = RandomForestClassifier(random_state=42)

        # Set up parameter grid for RandomizedSearchCV
        param_grid = {
            "n_estimators": [50, 60, 70, 100, 120, 150],  # Number of trees
            "max_depth": [None, 3, 4, 5, 6, 7],  # Depth of trees
            "min_samples_split": [2, 2, 4, 5, 6],  # Minimum samples to split a node
            "min_samples_leaf": [1, 2, 3, 4],  # Minimum samples at a leaf node
        }

        # Initialize RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=100,
            cv=5,
            verbose=2,
            random_state=1,
            n_jobs=-1,
            scoring="accuracy",
        )

        # Fit the model
        search.fit(X_train, y_train)

        # Get the best model
        best_model = search.best_estimator_

        # Make predictions on validation set
        y_pred = best_model.predict(X_valid)
        
        features = best_model.feature_names_in_
        joblib.dump(best_model, f"{MODEL_PATH}/{MODEL_NAME}_model.pkl")
        pd.DataFrame(search.cv_results_).to_csv(f"{CV_RESULTS_PATH}/cv_results_{MODEL_NAME}.csv", index=False)
    
    elif MODEL_NAME == "xgboost":
        # Initialize the model
        model = XGBClassifier(random_state=42)

        # Set up parameter grid for RandomizedSearchCV
        param_grid = {
                    'n_estimators': [80, 100, 120, 150],
                    'max_depth': [3, 4, 5, 6],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_weight': [1, 2, 3]
                }

        # Initialize RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=100,
            cv=5,
            verbose=2,
            random_state=1,
            n_jobs=-1,
            scoring="accuracy",
        )

        # Fit the model
        search.fit(X_train, y_train)

        # Get the best model
        best_model = search.best_estimator_

        # Make predictions on validation set
        y_pred = best_model.predict(X_valid)
        
        features = best_model.feature_names_in_
        joblib.dump(best_model, f"{MODEL_PATH}/{MODEL_NAME}_model.pkl")
        pd.DataFrame(search.cv_results_).to_csv(f"{CV_RESULTS_PATH}/cv_results_{MODEL_NAME}.csv", index=False)

    elif MODEL_NAME == "catboost":
        # Initialize the model
        model = CatBoostClassifier(random_state=42, eval_metric='MultiClass')

        # Set up parameter grid for RandomizedSearchCV
        # param_grid = {
        #     "depth": [4, 5, 6, 7, 8, 9, 10],  # Depth of trees
        #     "l2_leaf_reg": [4, 5, 6, 7, 8, 9, 10],  # Minimum samples at a leaf node
        #     "min_data_in_leaf": [1, 2, 3, 4],  # Minimum samples at a leaf node
        # }

        # # Initialize RandomizedSearchCV
        # search = RandomizedSearchCV(
        #     estimator=model,
        #     param_distributions=param_grid,
        #     n_iter=100,
        #     cv=5,
        #     verbose=2,
        #     random_state=1,
        #     n_jobs=-1,
        #     scoring="accuracy",
        # )

        # # Fit the model
        # search.fit(X_train, y_train)

        # # Get the best model
        # best_model = search.best_estimator_

        # # Make predictions on validation set
        # y_pred = best_model.predict(X_valid)
        
        # features = best_model.feature_names_in_
        # joblib.dump(best_model, f"{MODEL_PATH}/{MODEL_NAME}_model.pkl")
        # pd.DataFrame(search.cv_results_).to_csv(f"{CV_RESULTS_PATH}/cv_results_{MODEL_NAME}.csv", index=False)
        model.fit(X_train, y_train)
        joblib.dump(model, f"{MODEL_PATH}/{MODEL_NAME}_model.pkl")
        y_pred = model.predict(X_valid)
        features = X_train.columns
    
    elif MODEL_NAME == "lightgbm":
        # Initialize the model
        model = LGBMClassifier(random_state=42)

        # Set up parameter grid for RandomizedSearchCV
        param_grid = {
            "n_estimators": [80, 100, 120],  # Number of trees
            # "max_depth": [-1, 3, 4, 5, 6],  # Depth of trees
            "min_child_samples": [5, 10, 20],  # Minimum samples at a leaf node
        }

        # Initialize RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=100,
            cv=5,
            verbose=2,
            random_state=1,
            n_jobs=-1,
            scoring="accuracy",
        )

        # Fit the model
        search.fit(X_train, y_train)

        # Get the best model
        best_model = search.best_estimator_

        # Make predictions on validation set
        y_pred = best_model.predict(X_valid)
        
        features = best_model.feature_names_in_
        joblib.dump(best_model, f"{MODEL_PATH}/{MODEL_NAME}_model.pkl")
        pd.DataFrame(search.cv_results_).to_csv(f"{CV_RESULTS_PATH}/cv_results_{MODEL_NAME}.csv", index=False)
    

    # Log the results
    logging.info(f"Best parameters: {search.best_params_}")
    logging.info(f"Model saved to: {MODEL_PATH}/{MODEL_NAME}_model.pkl")
    logging.info(f"CV results saved to: {CV_RESULTS_PATH}/cv_results_{MODEL_NAME}.csv")
    logging.info(f"Features used to train model: {features}")

    valid_data["salary_category_pred"] = y_pred
    valid_data["salary_category_pred"] = valid_data["salary_category_pred"].map({0: "Low", 1: "Medium", 2: "High"})

    acc = accuracy_score(valid_data["salary_category"], valid_data["salary_category_pred"])
    f1_scoring = f1_score(valid_data["salary_category"], valid_data["salary_category_pred"], average="weighted")
    with Live("cv_results/", save_dvc_exp=True) as live:
        live.log_metric("accuracy", acc)
        live.log_metric("f1-score", f1_scoring)
        live.log_metric("selected_features", ", ".join(best_model.feature_names_in_))
        live.log_metric("best_params", str(search.best_params_))
    metrics = {
        "model": MODEL_NAME,
        "accuracy": acc,
        "f1-score": f1_scoring,
        "selected_features": ", ".join(best_model.feature_names_in_),
        "best_params": str(search.best_params_),
    }
    logging.info(f"Accuracy: {acc}, F1-score: {f1_scoring}")
    json.dump(obj=metrics, fp=open(f"{CV_RESULTS_PATH}/metrics.json", "w"), indent=4)

 
