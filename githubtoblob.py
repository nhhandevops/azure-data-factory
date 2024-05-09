import os
import requests
import sys
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Define your GitHub URL and the file path within the repository
train_url = "https://raw.githubusercontent.com/nhhandevops/azure-data-factory/main/mlopspythonscript/train_pipeline.py"
main_url = "https://raw.githubusercontent.com/nhhandevops/azure-data-factory/main/mlopspythonscript/aml-pipeline.py"
prepare_url = "https://raw.githubusercontent.com/nhhandevops/azure-data-factory/main/mlopspythonscript/prepare_pipeline.py"
data_url = "https://raw.githubusercontent.com/nhhandevops/azure-data-factory/main/test/data.csv"
# Define your Azure Blob Storage connection string and container name
account_name = "mloptestsa"
container_name = "mloptestcontainer"
directory_path = "traindata/mlops-scripts"

# Function to upload file to Azure Blob Storage
def upload_file_to_blob(file_url, container_client):
    # Get the file name from the URL
    file_name = os.path.basename(file_url)

    # Download the file from GitHub
    response = requests.get(file_url)
    if response.status_code == 200:
        # Upload the file to Blob Storage
        blob_client = container_client.get_blob_client(f"{directory_path}/{file_name}")
        blob_client.upload_blob(response.content, overwrite=True)
        print(f"File '{file_name}' uploaded to Blob Storage successfully.")
    else:
        print(f"Failed to download file from GitHub: {response.status_code}")

# Upload the file to Blob Storage
def main(connection_string):
    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Create a ContainerClient to interact with the container
    container_client = blob_service_client.get_container_client(container_name)
    upload_file_to_blob(train_url, container_client)
    upload_file_to_blob(prepare_url, container_client)
    upload_file_to_blob(main_url, container_client)
    upload_file_to_blob(data_url, container_client)


if __name__ == '__main__':
    arguments = sys.argv
    account_key = str(arguments[1])
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    main(connection_string)