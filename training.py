from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import ingestion
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'])
output_model_name = 'trainedmodel.pkl'
target_label = 'exited'
output_model_path = os.path.join(model_path, output_model_name)

##################Helper Functions
def load_training_data():
    logging.info("Loading training data...")
    df = pd.read_csv('/'.join([dataset_csv_path, ingestion.output_data_filename]))
    return df

#################Function for training the model
def train_model():
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    df = load_training_data()
    X = df.drop(columns=[target_label])
    # Select only numeric columns
    X = X.select_dtypes(include=[float, int])
    y = df[target_label]
    logging.info("Training model...")
    model.fit(X, y)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info(f"Saving model to {output_model_path}")
    joblib.dump(model, output_model_path)

if __name__ == "__main__":
    train_model()