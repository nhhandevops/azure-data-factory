#!/usr/bin/env python
# coding: utf-8

#from azureml.core.model import Model
import joblib
import pandas as pd
import sys
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, BlobSasPermissions

account_name = "mloptestsa"
container_name = "mloptestcontainer"
directory_name = "model_output"
model_name = "price_car_data.pkl"
output_file = "output.txt"
blob_name = f"{directory_name}/{model_name}"
output_name = f"{directory_name}/{output_file}"

# Load the model from Blob Storage
def load_model_from_blob_storage(connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(model_name, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    return joblib.load(model_name)

def init(connection_string):
    # model_path = Model.get_model_path(model_name="price_car_data.pkl")
    # model = joblib.load(model_path)
    model = load_model_from_blob_storage(connection_string)
    return model

def predict(model, connection_string):
    new_car = pd.DataFrame({'name': [2], 'year': [2015], 'km_driven': [80000]})
    predicted_price = model.predict(new_car)
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=output_name)
    
    str_output = f"Predicted selling price for the car : {predicted_price}"
    content_bytes = str_output.encode()
    blob_client.upload_blob(content_bytes, overwrite=True)

def main():
    arguments = sys.argv
    account_key = str(arguments[1])
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    model = init(connection_string)
    predict(model, connection_string)

if __name__ == '__main__':
    main()
