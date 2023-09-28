import pandas as pd
import pickle
import os
from sklearn import metrics
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = config['test_data_path'] 
output_model_path = config['output_model_path']

#################Function for model scoring
def score_model(eval_csv_path=None, model_path=None):
    eval_csv_path = eval_csv_path if eval_csv_path else os.path.join(test_data_path, "testdata.csv")
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    eval_df = pd.read_csv(eval_csv_path)
    features = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    X = eval_df[features]
    y = eval_df['exited']

    model_path = model_path if model_path else os.path.join(output_model_path, 'trainedmodel.pkl')
    model = pickle.load(open(model_path, 'rb'))

    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y, y_pred)

    with open(os.path.join(output_model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(f1_score))
    return f1_score

if __name__ == '__main__':
    score_model()