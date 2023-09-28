import os
import json
import logging

import ingestion
import training
import scoring
import deployment
import reporting


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(levelname)s %(message)s")
logger = logging.getLogger()


with open('config.json','r') as f:
    config = json.load(f)
input_folder_path = config['input_folder_path'] 
prod_deployment_path = config['prod_deployment_path']
output_folder_path = config['output_folder_path']

if not os.listdir(prod_deployment_path):
    # First-run
    ingestion.merge_multiple_dataframe()
    training.train_model()
    scoring.score_model()
    deployment.store_model_into_pickle()
else:
    ##################Check and read new data
    #first, read ingestedfiles.txt
    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
        ingested_files = set(f.read().split("\n"))
    
    csv_files = set([f for f in os.listdir(input_folder_path) if f.endswith(".csv")])

    non_ingested_files = csv_files - ingested_files

    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if non_ingested_files:
        logger.info("New files are detected, ingestion is running...")
        ingestion.merge_multiple_dataframe()

        ##################Checking for model drift
        #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
        with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
            old_score = float(f.read())
        
        new_score = scoring.score_model(eval_csv_path=os.path.join(output_folder_path, "finaldata.csv"),
                                        model_path=os.path.join(prod_deployment_path, "trainedmodel.pkl")
                                        )

        ##################Deciding whether to proceed, part 2
        #if you found model drift, you should proceed. otherwise, do end the process here
        is_drift = new_score < old_score

        if is_drift:
            ##################Re-deployment
            #if you found evidence for model drift, re-run the deployment.py script
            logger.info(f"Model drift is detected new: {new_score} < old: {old_score}, re-training and re-deployment are running...")
            training.train_model()
            deployment.store_model_into_pickle()

            ##################Diagnostics and reporting
            #run diagnostics.py and reporting.py for the re-deployed model
            os.system("python3 reporting.py")
            os.system("python3 apicalls.py")
        else:
            logger.info(f"No model drift is detected new: {new_score} >= old: {old_score}.")






