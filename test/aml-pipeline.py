from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import sys

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

    #-------------------------------------------------------------------------------------
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
    
    conda_path = generate_sas_blob_file_with_url(account_name,container_name, blob_name= " traindata/scripts/dependencies/conda.yaml")

    import os
    from azure.ai.ml.entities import Environment

    # dependencies_dir = "./dependencies"
    # os.makedirs(dependencies_dir, exist_ok=True)

    custom_env_name = "aml-mlops-test"

    custom_job_env = Environment(
        name=custom_env_name,
        description="Custom environment for MLOpstest Car Data",
        tags={"scikit-learn": "1.0.2"},
        conda_file=str(conda_path),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    custom_job_env = ml_client.environments.create_or_update(custom_job_env)

    print(
        f"Environment with name {custom_job_env.name} is registered to workspace, the environment version is {custom_job_env.version}"
    )

    #------------------------------------------------------------------------------------

    from azure.ai.ml import command
    from azure.ai.ml import Input   
    account_name = "mloptestsa"
    container_name = "mloptestcontainer"
    registered_model_name = "price_defaults_model"
    data_asset = ml_client.data.get(name="car_data", version="initial")

    job = command(
        inputs=dict(
            data=Input(
                type="uri_file",
                path=data_asset.path,
            ),
            registered_model_name=registered_model_name,
        ),
        code="./",  # location of source code
        compute="annhh1",
        command=f"python train_pipeline.py {TENANT_ID} {SUBSCRIPTION} {account_key} ",
        environment="aml-mlops-test@latest",
        display_name="price_default_prediction",
    )

    ml_client.create_or_update(job)



if __name__ == '__main__':
    #Get key
    arguments = sys.argv
    TENANT_ID = str(arguments[1])
    SUBSCRIPTION = str(arguments[2])
    account_key = str(arguments[3])
    main(TENANT_ID, SUBSCRIPTION, account_key)
