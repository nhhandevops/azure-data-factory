from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, ClientSecretCredential 
import sys

def main(CLIENT_ID, TENANT_ID, CLIENT_SECRET, SUBSCRIPTION, account_key):
    RESOURCE_GROUP="rg-ml-dev"
    WS_NAME="ws-ml-dev"

    #authenticate
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        # This will open a browser page for
        credential = ClientSecretCredential(client_id=CLIENT_ID, tenant_id=TENANT_ID, client_secret = CLIENT_SECRET)
        #credential = InteractiveBrowserCredential()

    # Get a handle to the workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WS_NAME,
    )

    #-------------------------------------------------------------------------------------
    #Configure compute instance provisioning
    from azure.ai.ml.entities import ComputeInstance
    from azure.ai.ml.constants import TimeZone
    from azure.ai.ml.entities import ComputeInstance, ComputeSchedules, ComputeStartStopSchedule, RecurrenceTrigger, RecurrencePattern

    ins_name="mlops-instance"
    ins_size="Standard_DS1_v2"

    #Schedule
    ci_start_time = "2024-05-08T08:30:00" #specify your start time in the format yyyy-mm-ddThh:mm:ss

    rec_trigger = RecurrenceTrigger(
        start_time=ci_start_time, 
        time_zone=TimeZone.SE_ASIA_STANDARD_TIME, 
        frequency="week", interval=1, 
        schedule=RecurrencePattern(week_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], hours=10, minutes=23))
    
    myschedule = ComputeStartStopSchedule(trigger=rec_trigger, action="start")
    com_sch = ComputeSchedules(compute_start_stop=[myschedule])

    # Create compute instance
    compute_instance_client = ComputeInstance(
        name=ins_name, 
        size=ins_size,
        description="Compute instance for my service principal",
        schedules=com_sch
    )

    ml_client.begin_create_or_update(compute_instance_client).result()

    #------------------------------------------------------------------------------------

    #Create azure machine learning components
    #DATA PREPARATION COMPONENTS
    from azure.ai.ml import command
    from azure.ai.ml import Input, Output

    data_prep_component = command(
        outputs=dict(
            dataset=Output(type="uri_folder", mode="rw_mount"),
        ),
        name="data_prep_credit_defaults",
        display_name="Data preparation for training",
        description="reads from blob, filter and save back to blob",
        # The source folder of the component
        code="./",
        command= "python prepare_pipeline.py --dataset ${{outputs.dataset}} --subscription " + SUBSCRIPTION,
        environment="aml-mlops-test@latest",
    )
    data_prep_component = ml_client.create_or_update(data_prep_component.component)

    #TRAIN DATA COMPONENT
    train_component = command(
        inputs=dict(
            data=Input(type="uri_folder"),
        ),
        name="train_car_data",
        display_name="Train and Predict data",
        description="get model and predict sample",
        # The source folder of the component
        code="./",
        command="python train_pipeline.py --data ${{inputs.data}} --subscription " + SUBSCRIPTION,
        environment="aml-mlops-test@latest",
    )
    train_component = ml_client.create_or_update(train_component.component)

    #------------------------------------------------------------------------------------

    from azure.ai.ml import command
    from azure.ai.ml import dsl, Input, Output
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
 
    # registered_model_name = "price_defaults_model"
    # data_asset = ml_client.data.get(name="car_data", version="initial")

    # job = command(
    #     inputs=dict(
    #     data=Input(path=data_asset.path,
    #           type=AssetTypes.URI_FILE,
    #           mode=InputOutputModes.RO_MOUNT
    #           ),
    #         registered_model_name=registered_model_name,
    #     ),
    #     code="./",  # location of source code
    #     compute=ins_name,
    #     command=f"python train_pipeline.py {CLIENT_ID} {SUBSCRIPTION} {account_key} ",
    #     environment="aml-mlops-test@latest",
    #     display_name="price_default_prediction",
    # )
    @dsl.pipeline(
        compute=ins_name,  # "serverless" value runs pipeline on serverless compute
        description="test mlops python script pipeline",
    )
    def credit_defaults_pipeline():
        # using data_prep_function like a python call
        data_prep_job = data_prep_component()

        # using train_func like a python call
        train_job = train_component(data = data_prep_job.outputs.dataset)

    # Let's instantiate the pipeline with the parameters of our choice
    pipeline = credit_defaults_pipeline()
    # submit the pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline,
        experiment_name="test-mlops-pipeline"
    )
    ml_client.jobs.stream(pipeline_job.name)
    #ml_client.create_or_update(job)

    #----------------------------------------------------------------------------------------


if __name__ == '__main__':
    #Get key
    arguments = sys.argv
    CLIENT_ID = str(arguments[1])
    TENANT_ID = str(arguments[2])
    CLIENT_SECRET= str(arguments[3])
    SUBSCRIPTION = str(arguments[4])
    account_key = str(arguments[5])
    main(CLIENT_ID, TENANT_ID, CLIENT_SECRET, SUBSCRIPTION, account_key)
