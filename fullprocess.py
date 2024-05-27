

import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import os
import json
import app_logger
import apicalls

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

logger = app_logger.Logger

source_data_path = os.path.join(config["input_folder_path"])

def get_ingested_files():
    with open(ingestion.ingested_records_filepath, 'r') as fr:
        ingested_files = fr.readlines()
    return ingested_files

def run_all():
    ##################Check and read new data
    #first, read ingestedfiles.txt
    ingested_files = get_ingested_files()
    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = os.listdir(source_data_path)
    new_source_files = set(source_files) - set(ingested_files)

    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if not new_source_files:
        return

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    old_f1_score = float(deployment.get_prod_f1_score())
    ingestion.ingest_data()
    new_f1_score = float(scoring.score_model(ingestion.output_dataframe_filepath, use_production=True))
    logger.info(f"Model f1 scores:\n\tF1 Score with Old Data: {old_f1_score}\n\tF1 Score with New Data: {new_f1_score}")
    has_model_drift = new_f1_score < old_f1_score

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if not has_model_drift:
        logger.info("Did not find model drift, not deploying model.")
        return

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    logger.info("Drift found, training and deploying new model...")
    training.train_model()
    deployment.deploy_model()

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    reporting.generate_model_report()
    diagnostics.run_diagnostics()
    apicalls.call_and_record_api()

if __name__ == "__main__":
    run_all()





