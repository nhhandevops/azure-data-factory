import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, ClientSecretCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import sys

def main():
    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription", type=str, help="subscription")
    parser.add_argument("--dataset", type=str, help="path to train data")

    args = parser.parse_args()
    SUBSCRIPTION = args.subscription

    # #--------------------------------------------------------------------------------------------

    RESOURCE_GROUP="rg-ml-dev"
    WS_NAME="ws-ml-dev"
    account_name = "mloptestsa"
    container_name = "mloptestcontainer"
    
    #credential = ClientSecretCredential(client_id=CLIENT_ID, tenant_id=TENANT_ID, client_secret = CLIENT_SECRET)
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    #credential.get_token("https://management.azure.com/.default")

    # Get a handle to the workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WS_NAME
    )

    #--------------------------------------------------------------

    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, BlobSasPermissions
    from datetime import datetime, timedelta

    # def generate_sas_blob_file_with_url (account_name, container_name, blob_name):
    #     sas = generate_blob_sas(account_name = account_name,
    #                                 container_name = container_name,
    #                                 blob_name = blob_name,
    #                                 account_key=account_key,
    #                                 permission=BlobSasPermissions(read=True),
    #                                 expiry=datetime.now() + timedelta(hours=1))

    #     sas_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + blob_name + '?' + sas
    #     file_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + blob_name
    #     return sas_url, file_url

    blob_name= "traindata/car_data.csv"
    test_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + blob_name
    # set the version number of the data asset
    # change to the new version if want to update the URI
    v1 = "1"

    my_data = Data(
        name="car_data",
        version=v1,
        description="Car data",
        path=test_url,
        type=AssetTypes.URI_FILE
    )

    ## create data asset if it doesn't already exist:
    try:
        data_asset = ml_client.data.get(name="car_data", version=v1)
        print(
            f"Data asset already exists. Name: {my_data.name}, version: {my_data.version}"
        )
    except:
        ml_client.data.create_or_update(my_data)
        print(f"Data asset created. Name: {my_data.name}, version: {my_data.version}")

    #--------------------------------------------------------------------------------------------
    import pandas as pd

    data_asset = ml_client.data.get(name="car_data", version=v1)
    #print(f"Data asset URI: {data_asset}")

    # read into pandas - note that you will see 2 headers in your data frame - that is ok, for now

    df = pd.read_csv(data_asset.path, header=1)

    #---------------------------------------------------------------------------------------------
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

    df_filter = filter_data(df)
    df_filter.to_csv(os.path.join(args.dataset, "data.csv"),index=False)

#--------------------------------------------------------------------------------------------------------------
    
    # from azure.storage.blob import BlobServiceClient
    # # authenticate with Blob Storage
    # account_name = "mloptestsa"
    # container_name = "mloptestcontainer"
    # connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"

    # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    # container_client = blob_service_client.get_container_client(container_name)

    # new_data = df_filter.to_csv()

    # blob_name = "filter_car_data.csv"
    # blob_client = container_client.get_blob_client(blob=f"model_output/{blob_name}")
    
    # blob_client.upload_blob(new_data, overwrite=True)

    # blob_name= "model_output/filter_car_data.csv"
    # test_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + blob_name


    #---------------------------------------------------------------------------------------------------
    
    

if __name__ == '__main__':
    #Get key
    # arguments = sys.argv
    # CLIENT_ID = str(arguments[2])
    # TENANT_ID = str(arguments[3])
    # CLIENT_SECRET= str(arguments[4])
    # SUBSCRIPTION = str(arguments[5])
    # account_key = str(arguments[6])
    main()