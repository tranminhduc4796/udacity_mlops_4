
import pandas as pd
import pickle
import numpy as np
import timeit
import os
import json

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_folder_path = os.path.join(config['output_folder_path'])

##################Function to get model predictions
def model_predictions(test_csv_path=None):
    #read the deployed model and a test dataset, calculate predictions
    test_csv_path = test_csv_path if test_csv_path else os.path.join(test_data_path, "testdata.csv")
    
    test_df = pd.read_csv(test_csv_path)
    features = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    X = test_df[features]
    y = test_df['exited']

    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))
    y_pred = model.predict(X)

    return y_pred

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))

    return {"mean": df.mean().to_dict(), "median": df.median().to_dict(), "std": df.std().to_dict()}

##################Function to get missing data precent
def dataframe_missing_percent():
    df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    miss_percent = (df.isnull().sum()/len(df)).to_dict()
    return miss_percent

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    os.system('python3 training.py')
    training_time = timeit.default_timer() - start_time

    return {'ingestion_time': ingestion_time, 'training_time': training_time}

##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    outdateds = os.popen('pip list --outdated').read()
    return outdateds


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
