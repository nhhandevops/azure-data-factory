from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, ClientSecretCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import sys

#--------------------------------------------------------------
    #Get data from the previous step
import os
def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

def main():
    #get SUBSCRIPTION
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--subscription", type=str, help="subscription")
    parser.add_argument("--data", type=str, help="path to train data")

    args = parser.parse_args()
    SUBSCRIPTION = args.subscription

    RESOURCE_GROUP="rg-ml-dev"
    WS_NAME="ws-ml-dev"

    credential = DefaultAzureCredential()
    #credential = ClientSecretCredential(client_id=CLIENT_ID, tenant_id=TENANT_ID, client_secret = CLIENT_SECRET)

    # Get a handle to the workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WS_NAME
    )
    
    #--------------------------------------------------------------------------------------------

    import pandas as pd

    data =  select_first_file(args.data)

    # read into pandas - note that you will see 2 headers in your data frame - that is ok, for now

    df = pd.read_csv(data)


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

    data, test_validation = split_data(df)
    linear_model = train_model(data)
    mse = get_model_metrics(linear_model, data, test_validation)

    print("Test loss:", mse)

if __name__ == '__main__':
    #Get key
    # arguments = sys.argv
    # CLIENT_ID = str(arguments[1])
    # TENANT_ID = str(arguments[2])
    # CLIENT_SECRET= str(arguments[3])
    # SUBSCRIPTION = str(arguments[4])
    # account_key = str(arguments[5])
    main()