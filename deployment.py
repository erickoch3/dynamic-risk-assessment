from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import training
import scoring
import ingestion
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
ingested_data_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
prod_f1_score_filepath = os.path.join(prod_deployment_path, scoring.f1_score_filename)

def get_prod_f1_score():
    with open(prod_f1_score_filepath, 'r') as fr:
        return fr.read()

####################Define Helpers
def copy_file(src, dst):
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            fdst.write(fsrc.read())

####################function for deployment
def deploy_model():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    trained_model_path = os.path.join(output_model_path, training.output_model_name)
    latest_score_path = os.path.join(output_model_path, scoring.f1_score_filename)
    ingested_files_path = os.path.join(ingested_data_path, ingestion.output_ingestedrecord_filename)
    
    # Ensure the production deployment directory exists
    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)
    
    # Define destination paths
    for filepath_to_copy in trained_model_path, latest_score_path, ingested_files_path:
        output_path = os.path.join(prod_deployment_path, os.path.basename(filepath_to_copy))
        copy_file(filepath_to_copy, output_path)
        
if __name__ == "__main__":
    deploy_model()