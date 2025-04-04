import argparse
from dvclive import Live
import joblib
import json
import numpy as np
import os
import logging
import pandas as pd 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from utils import load_params

log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'train_evaluate.log'),
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

    DATA_PATH = params["train_evaluate"]["data_path"]
    MODEL_PATH = params["train_evaluate"]["model_path"]
    CV_RESULTS_PATH = params["train_evaluate"]["cv_results_path"]

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(CV_RESULTS_PATH, exist_ok=True)

    # Read train and validation data
    train_data = pd.read_csv(f"{DATA_PATH}/training_data_fs.csv")
    train_data['salary_category_num'] = train_data['salary_category'].map({'High': 2, 'Medium': 1, 'Low':0})
    valid_data = pd.read_csv(f"{DATA_PATH}/validation_data_fs.csv")
    valid_data['salary_category_num'] = valid_data['salary_category'].map({'High': 2, 'Medium': 1, 'Low':0})

    # Separate features and target variable
    X_train = train_data.drop(columns=["salary_category_num", "salary_category"])
    y_train = train_data["salary_category_num"]
    
    X_valid = valid_data.drop(columns=["salary_category_num", "salary_category"])
    y_valid = valid_data["salary_category_num"]

    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Set up parameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [50, 60, 70, 100, 120, 150],        # Number of trees
        'max_depth': [None, 3,4, 5, 10],           # Depth of trees
        'min_samples_split': [2, 5, 10],       # Minimum samples to split a node
        'min_samples_leaf': [1, 2, 4],         # Minimum samples at a leaf node
    }

    # Initialize RandomizedSearchCV
    search = RandomizedSearchCV(estimator=model,
                                param_distributions=param_grid,
                                n_iter=100,
                                cv=5,
                                verbose=2,
                                random_state=1,
                                n_jobs=-1,
                                scoring='accuracy')

    # Fit the model
    search.fit(X_train, y_train)

    # Get the best model
    best_model = search.best_estimator_

    # Make predictions on validation set
    y_pred = best_model.predict(X_valid)

    joblib.dump(best_model, f"{MODEL_PATH}/rf_model.pkl")
    pd.DataFrame(search.cv_results_).to_csv(f"{CV_RESULTS_PATH}/rf_model_results.csv", index=False)

    with Live("cv_results/", save_dvc_exp=True) as live:
        live.log_metric("features", ", ".join(best_model.feature_names_in_))
        live.log_metric("best_params", str(search.best_params_))
    
    # Log the results
    logging.info(f"Best parameters: {search.best_params_}")
    logging.info(f"Model saved to: ../models/rf_model.pkl")
    logging.info(f"CV results saved to: ../cv_results/rf_model_results.csv")
    logging.info(f"Features used to train model: {best_model.feature_names_in_}")
    
    valid_data['salary_category_pred'] = y_pred
    valid_data['salary_category_pred'] = valid_data['salary_category_pred'].map({0: 'Low', 1: 'Medium', 2: 'High'})

    acc = accuracy_score(valid_data['salary_category'], valid_data['salary_category_pred'])
    f1_scoring = f1_score(valid_data['salary_category'], valid_data['salary_category_pred'], average='weighted')
    with Live("cv_results/", save_dvc_exp=True) as live:
        live.log_metric("accuracy", acc)
        live.log_metric("f1-score", f1_scoring)
    metrics = {'accuracy': acc, "f1-score": f1_scoring}
    logging.info(f"Accuracy: {acc}, F1-score: {f1_scoring}")
    json.dump(
        obj=metrics,
        fp=open(f"{CV_RESULTS_PATH}/metrics.json", 'w'),
        indent=4
    )