from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import joblib
import training
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import app_logger

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

logger = app_logger.Logger

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])
test_data_dir_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
testdata_name = 'testdata.csv'
f1_score_filename = 'latestscore.txt'
score_filepath = os.path.join(output_model_path, f1_score_filename)
test_data_filepath = os.path.join(test_data_dir_path, testdata_name)

#################Utility Functions
def load_staged_model():
    model_path = os.path.join(output_model_path, training.output_model_name)
    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    logger.info("Successfully loaded model!")
    return model

def load_production_model():
    model_path = os.path.join(prod_deployment_path, training.output_model_name)
    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    logger.info("Successfully loaded model!")
    return model

def load_test_data(dataset_filepath):
    
    # Get the directory of the current script's parent folder
    script_parent_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path
    if dataset_filepath.startswith('/'):
        dataset_filepath = dataset_filepath.lstrip('/')

    logger.info(f"Loading test data {dataset_filepath}...")
    dataset_filepath = os.path.join(script_parent_dir, dataset_filepath)
    # Load the data
    return pd.read_csv(dataset_filepath)

def load_X_y_test_data(dataset_filepath):
    test_data = load_test_data(dataset_filepath)
    logger.info("Converting to X and y...")
    # Extract features and labels
    X_test = test_data.drop(columns=[training.target_label])
    y_test = test_data[training.target_label]
    # Select only numeric columns
    X_test = X_test.select_dtypes(include=[float, int])
    return X_test, y_test

def make_predictions(model, dataset_filepath):
    # Load the Model and the Test Data
    try:
        X_test, _ = load_X_y_test_data(dataset_filepath)
    except FileNotFoundError:
        logger.error(f"File not found on server at {dataset_filepath}")
        return None
    
    # Make predictions
    y_pred = model.predict(X_test)
    return y_pred

#################Function for model scoring
def score_model(data_filepath, use_production=False):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    try:
        _, y_test = load_X_y_test_data(data_filepath)
    except FileNotFoundError as e:
        logger.error(f"No file found at path {data_filepath}. Cannot score.")
        raise e
    if use_production:
        model = load_production_model()
    else:
        model = load_staged_model()
    y_pred = make_predictions(model, data_filepath)

    # Calculate the F1 score
    logger.info("Scoring model...")
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    logger.info(f"Model has f1 score: {f1}")
    
    logger.info(f"Writing score to {score_filepath}")
    with open(score_filepath, 'w') as fw:
        fw.write(str(f1))
    
    return f1

if __name__ == "__main__":
    score_model(test_data_filepath)