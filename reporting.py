import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
output_model_path = config['output_model_path']


##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    features = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    X = df[features]
    y = df['exited']

    model = pickle.load(open(os.path.join(output_model_path, 'trainedmodel.pkl'), 'rb'))

    y_pred = model.predict(X)
    confusion_mat = metrics.confusion_matrix(y, y_pred)
    plt.figure(figsize=(9, 9))
    sns.heatmap(confusion_mat, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    out_filename = 'confusionmatrix.png'
    if os.path.exists(os.path.join(output_model_path, out_filename)):
        out_filename = 'confusionmatrix2.png'
    plt.savefig(os.path.join(output_model_path, out_filename))

if __name__ == '__main__':
    score_model()
