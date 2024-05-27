import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import scoring
import app_logger

logger = app_logger.Logger

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
confusion_matrix_filename = 'confusionmatrix.png'
confusion_matrix_filepath = os.path.join(output_model_path, confusion_matrix_filename)

##############Helper Functions
def build_confusion_matrix_plot(y_true, y_pred):
    logger.info("Generating a confusion matrix plot for the model...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    return plt

##############Function for reporting
def generate_model_report():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace    
    _, y_true = scoring.load_X_y_test_data(scoring.test_data_filepath)
    model = scoring.load_production_model()
    y_pred = scoring.make_predictions(model, scoring.test_data_filepath)

    plt = build_confusion_matrix_plot(y_true, y_pred)
    logger.info(f"Saving plot to {confusion_matrix_filepath}")
    plt.savefig(confusion_matrix_filepath)
    plt.close()  # Close the plot to free memory


if __name__ == '__main__':
    generate_model_report()
