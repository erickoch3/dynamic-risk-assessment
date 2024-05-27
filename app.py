from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import diagnostics
import scoring



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])

prediction_model = None

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    # Accessing the JSON data sent in the POST request
    data = request.json
    dataset_filepath = data.get("dataset_filepath")

    # Ensure that dataset_filepath is provided
    if not dataset_filepath:
        return jsonify({"error": "dataset_filepath is required"}), 400

    # Call the prediction function
    try:
        model = scoring.load_production_model()
        predictions = diagnostics.model_predictions(model, dataset_filepath)
        
        # Ensure predictions are JSON serializable
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.to_dict(orient='records')
        elif isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        
    except FileNotFoundError:
        return jsonify({"error": f"{dataset_filepath} not found on server"}), 400
    
    # Return predictions as JSON response
    return jsonify(predictions)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    f1_score = scoring.score_model(scoring.test_data_filepath)
    scoring_data = {
        "f1_score": f1_score
    }
    return jsonify(scoring_data)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    summary_stats = diagnostics.dataframe_summary()
    return jsonify(summary_stats)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnosis():
    #check timing and percent NA values
    process_times = diagnostics.execution_time()
    na_value_percentage = diagnostics.check_missing_data()
    outdated_dependencies = diagnostics.outdated_packages_list()
    # Convert the outdated_dependencies DataFrame to a list of dictionaries
    outdated_dependencies_list = outdated_dependencies.to_dict(orient='records')
    # Combine diagnostics into a structured dictionary
    diagnostics_data = {
        "process_times": process_times,
        "na_value_percentage": na_value_percentage,
        "outdated_dependencies": outdated_dependencies_list
    }

    # Return the diagnostics data as a JSON response
    return jsonify(diagnostics_data)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
