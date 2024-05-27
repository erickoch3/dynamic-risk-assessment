import requests
import json
import os
import app_logger

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

logger = app_logger.Logger

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

output_dir_path = config['output_model_path']
apireturns_filename = "apireturns.txt"
apireturns_filepath = os.path.join(output_dir_path, apireturns_filename)


def call_and_record_api():
    #Call each API endpoint and store the responses
    try:
        predictions = requests.post(URL + "prediction", json={"dataset_filepath": "/testdata/testdata.csv"})
        score = requests.get(URL + "scoring")
        summarystats = requests.get(URL + "summarystats")
        diagnostics = requests.get(URL + "diagnostics")
        # Combine all API responses as text
        responses = "\n".join([
            predictions.text,
            score.text,
            summarystats.text,
            diagnostics.text
        ])

        #write the responses to your workspace
        with open(apireturns_filepath, 'w') as fw:
            fw.write(responses)
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Failed to connect to the API. Is it running?\nFull error: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Request timed out: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"An error occurred: {req_err}")


if __name__ == "__main__":
    call_and_record_api()