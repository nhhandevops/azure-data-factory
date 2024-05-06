from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import sys

SUBSCRIPTION="5f5c1301-d508-45a5-a13c-b5190ea4cf8a"
RESOURCE_GROUP="rg-ml-dev"
WS_NAME="ws-ml-dev"
account_name = "mloptestsa"
container_name = "mloptestcontainer"

def main(TENANT_ID, SUBSCRIPTION, account_key):
    # authenticate
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        # This will open a browser page for
        credential = InteractiveBrowserCredential(tenant_id=TENANT_ID)

    # Get a handle to the workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WS_NAME,
    )

    #--------------------------------------------------------------

    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, BlobSasPermissions
    from datetime import datetime, timedelta

    def generate_sas_blob_file_with_url (account_name, container_name, blob_name):
        sas = generate_blob_sas(account_name = account_name,
                                    container_name = container_name,
                                    blob_name = blob_name,
                                    account_key=account_key,
                                    permission=BlobSasPermissions(read=True),
                                    expiry=datetime.now() + timedelta(hours=1))

        sas_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + blob_name + '?' + sas
        return sas_url

    my_path = generate_sas_blob_file_with_url(account_name,container_name, blob_name= "traindata/car_data.csv")
    # set the version number of the data asset
    v1 = "initial"

    my_data = Data(
        name="car_data",
        version=v1,
        description="Car data",
        path=my_path,
        type=AssetTypes.URI_FILE,
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
    #print(f"Data asset URI: {data_asset.path}")

    # read into pandas - note that you will see 2 headers in your data frame - that is ok, for now

    df = pd.read_csv(data_asset.path)

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
        df_filtered_columns_rows_header = df_filtered_columns_rows.iloc[1:]
        return df_filtered_columns_rows_header

    df_filter = filter_data(df)

    #TRAIN SCRIPT
    #---------------------------------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import accuracy_score

    def split_data(df_filtered_columns_rows):
        X = df_filtered_columns_rows.drop(columns=['selling_price']).values
        y = df_filtered_columns_rows['selling_price'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data = {"train": {"X": X_train, "y": y_train},
                "test": {"X": X_test, "y": y_test}}
        
        return data, y_test

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

    data, test_validation = split_data(df_filter)
    linear_model = train_model(data)
    mse = get_model_metrics(linear_model, data, test_validation)

    print("Test loss:", mse)

if __name__ == '__main__':
    #Get key
    arguments = sys.argv
    TENANT_ID = str(arguments[1])
    SUBSCRIPTION = str(arguments[2])
    account_key = str(arguments[3])
    main(TENANT_ID, SUBSCRIPTION, account_key)