import os
import requests
import json


#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

with open('config.json','r') as f:
    config = json.load(f) 
output_model_path = config['output_model_path']

#Call each API endpoint and store the responses
response1 = requests.post(URL + "prediction", json={"csv_path": "testdata/testdata.csv"})
response2 = requests.get(URL + "scoring")
response3 = requests.get(URL + "summarystats")
response4 = requests.get(URL + "diagnostics")

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace
out_filename = 'apireturns.txt'
if os.path.exists(os.path.join(output_model_path, out_filename)):
        out_filename = 'apireturns2.txt'
with open(os.path.join(output_model_path, out_filename), 'w') as f:
    for response in responses:
        f.write(response.text)


