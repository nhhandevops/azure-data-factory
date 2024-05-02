#!/usr/bin/env python
# coding: utf-8

#from azureml.core.model import Model
import joblib
import pandas as pd
import sys
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, BlobSasPermissions
from azure.identity import DefaultAzureCredential
from datetime import datetime, timedelta
import requests
from io import BytesIO


credential = DefaultAzureCredential()
account_name = "mloptestsa"
container_name = "mloptestcontainer"
directory_name = "model_output"
model_name = "price_car_data.pkl"
temp_model_name = "price_car_data.pkl"
output_file = "output.txt"
blob_name = f"{directory_name}/{model_name}"
output_name = f"{directory_name}/{output_file}"

def get_into_container_blob_storage (connection_string, container_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    return container_client


def generate_sas_blob_file_with_url (account_name, container_name, blob_name):
    sas = generate_blob_sas(account_name = account_name,
                                container_name = container_name,
                                blob_name = blob_name,
                                account_key=account_key,
                                permission=BlobSasPermissions(read=True),
                                expiry=datetime.now() + timedelta(hours=1))

    sas_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + blob_name + '?' + sas
    return sas_url
def load_model_from_sas_url(sas_url):
    # Send a GET request to the SAS URL to download the file content
    response = requests.get(sas_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Load the file content into a BytesIO object
        file_content = BytesIO(response.content)
        
        # Load the model using joblib
        loaded_model = joblib.load(file_content)
        
        return loaded_model
    else:
        print("Failed to download file from SAS URL. Status code:", response.status_code)
        return None

# Load the model from Blob Storage
# def load_model_from_blob_storage(connection_string):
#     blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#     blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
#     with open(temp_model_name, "wb") as download_file:
#         download_file.write(blob_client.download_blob().readall())
#     return joblib.load(temp_model_name)
    

# def init(connection_string):
#     # model_path = Model.get_model_path(model_name="price_car_data.pkl")
#     # model = joblib.load(model_path)
#     model = load_model_from_blob_storage(connection_string)
#     return model

def predict(model, connection_string):
    new_car = pd.DataFrame({'name': [2], 'year': [2015], 'km_driven': [80000]})
    predicted_price = model.predict(new_car)
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=output_name)
    
    str_output = f"Predicted selling price for the car : {predicted_price}"
    content_bytes = str_output.encode()
    blob_client.upload_blob(content_bytes, overwrite=True)

def main(connection_string):
    #model = init(connection_string)
    sas_url = generate_sas_blob_file_with_url(account_name, container_name, blob_name)
    model = load_model_from_sas_url(sas_url)
    predict(model, connection_string)

if __name__ == '__main__':
    arguments = sys.argv
    account_key = str(arguments[1])
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    main(connection_string)
