import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    csv_files = [f for f in os.listdir(input_folder_path) if f.endswith(".csv")]

    df = pd.DataFrame()
    for f in csv_files:
        df = df.append(pd.read_csv(os.path.join(input_folder_path, f)))
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

    # Record files to be merged together
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.write("\n".join(csv_files))

if __name__ == '__main__':
    merge_multiple_dataframe()
