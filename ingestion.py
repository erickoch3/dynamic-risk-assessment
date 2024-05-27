import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import app_logger

logger = app_logger.Logger

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
output_data_filename = 'finaldata.csv'
output_ingestedrecord_filename = 'ingestedfiles.txt'
ingested_records_filepath = os.path.join(output_folder_path, output_ingestedrecord_filename)
output_dataframe_filepath = os.path.join(output_folder_path, output_data_filename)

#############Helper Functions
def get_all_dataframes(folder_path):
    logger.info(f"Retrieving all dataframes from CSV's in {folder_path}...")
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = [pd.read_csv(os.path.join(folder_path, f)) for f in csv_files]
    return csv_files, dataframes
        
def outer_merge_dataframes(dataframes):
    logger.info("Performing outer merge on dataframes...")
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, how='outer')
    logger.info("Created merged dataframe.\n")
    merged_df.info()
    logger.info(f"Dataframe Summary:\n{merged_df.describe()}\n")
    logger.info(f"Dataframe Preview:\n{merged_df.head()}\n")
    return merged_df

def write_ingested_files(file_list):
    logger.info(f"Logging ingested filenames to {ingested_records_filepath}...")
    with open(ingested_records_filepath, 'w') as file:
        for filename in file_list:
            file.write(filename + '\n')

def write_to_csv(dataframe, output_path):
    logger.info(f"Writing new dataframe to {output_path}...")
    dataframe.to_csv(output_path, index=False)

#############Function for data ingestion
def ingest_data():
    #check for datasets, compile them together, and write to an output file
    ingested_files, dataframes = get_all_dataframes(input_folder_path)
    merged_df = outer_merge_dataframes(dataframes)
    output_df = merged_df.drop_duplicates()
    write_ingested_files(ingested_files)
    write_to_csv(output_df, output_dataframe_filepath)

if __name__ == '__main__':
    ingest_data()
