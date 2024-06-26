#!/usr/bin/env python
# coding: utf-8

#test agin

#Configure the access to the AML workspace
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import sys

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import os

#Library for train
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# authenticate
credential = DefaultAzureCredential()
SUBSCRIPTION=""
RESOURCE_GROUP="rg-ml-dev"
WS_NAME="ws-ml-dev"

# # Get a handle to the workspace
# ml_client = MLClient(
#     credential=credential,
#     subscription_id=SUBSCRIPTION,
#     resource_group_name=RESOURCE_GROUP,
#     workspace_name=WS_NAME,
# )

# ws = ml_client.workspaces.get(WS_NAME)

account_name = "mloptestsa"
container_name = "mloptestcontainer"

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

# Get path of data file in Blob storage
def get_dataframe_blob_file(connection_string, container_name):
    # load file from blob
    container_client = get_into_container_blob_storage (connection_string, container_name)

    blob_target = []
    for blob_i in container_client.list_blobs():
        if blob_i.name == "traindata/car_data.csv":
            blob_target = blob_i.name
        
    #generate a shared access signature for each blob file
    sas_url = generate_sas_blob_file_with_url(account_name, container_name, blob_target)

    #read data file from blob url
    df = pd.read_csv(sas_url, skiprows=1)
    return df
    
#Filter & transform data frame - remove unnecessary columns and null value in rows
def filter_data(df): 
    #Remove unselected column
    df.columns = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
    columns_drop = ['fuel','seller_type','owner','transmission']
    df_filtered_columns = df.drop(columns=columns_drop)

    #Get the column "name" to numeric type
    df_filtered_columns['name'], uniques = pd.factorize(df_filtered_columns['name'])
    
    #Delete missing value row
    df_filtered_columns_rows = df_filtered_columns.dropna()
    return df_filtered_columns_rows

#Split data into train data and test data
def split_data(df_filtered_columns_rows):
    X = df_filtered_columns_rows.drop(columns=['selling_price']).values
    y = df_filtered_columns_rows['selling_price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    
    return data, y_test

#Train the model, return the model
def train_model(data):
    car_model = LinearRegression()
    car_model.fit(data["train"]["X"], data["train"]["y"])
    return car_model

#Evaluate the metrics for the model
def get_model_metrics(model, data, test_validation):
    preds = model.predict(data["test"]["X"])
    mse = mean_squared_error(preds, test_validation)
    metrics = {"mse": mse}
    return metrics

#save trained model to file for SCORE model to gain prediction
def save_model(model):
    model_name = "price_car_data.pkl"
    joblib.dump(model, filename=model_name)
    container_client = get_into_container_blob_storage(connection_string, container_name)
    blob_client = container_client.get_blob_client(blob=f"model_output/{model_name}")
    with open(model_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

def main(connection_string, container_name):
    # Load Data Frame form Blob storage
    df = get_dataframe_blob_file(connection_string, container_name)
    

    # Split Data into Training and Validation Sets
    df_filter = filter_data(df)
    data, test_validation = split_data(df_filter)

    # Train Model on Training Set
    
    linear_model = train_model(data)

    # Validate Model on Validation Set
    metrics = get_model_metrics(linear_model, data, test_validation)

    # Save Model
    save_model(linear_model)
    print(metrics)

if __name__ == '__main__':
    # Accessing command-line arguments
    arguments = sys.argv
    account_key = str(arguments[1])
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    main(connection_string, container_name)
