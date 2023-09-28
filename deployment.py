from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(config['output_folder_path'])
output_model_path = config['output_model_path']
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copy2(os.path.join(output_model_path, 'trainedmodel.pkl'), prod_deployment_path)
    shutil.copy2(os.path.join(output_model_path, 'latestscore.txt'), prod_deployment_path)
    shutil.copy2(os.path.join(output_folder_path, 'ingestedfiles.txt'), prod_deployment_path)

if __name__ == '__main__':
    store_model_into_pickle()

        

