#############################################################
#
#   For memory reason, let's work for now with parquet/INT8 data
#
############################################################
# %%  Load package
#!pip install numerapi // 
from pathlib import Path
import numerapi
# %% Create instance of NumerAPI
napi = numerapi.NumerAPI()

#List available datasets
napi.list_datasets()


# %% Download training and validation data
train_pq_path = "data/numerai_training_data_int8.parquet"
val_pq_path = "data/numerai_validation_data_int8.parquet"
# download the data if not already done
napi.download_dataset("numerai_training_data_int8.parquet", train_pq_path)
napi.download_dataset("numerai_validation_data_int8.parquet", val_pq_path)


# %%# Download all current data round data
CURRENT_ROUND = napi.get_current_round()

#Look for for parquet / int8 data files
for file in napi.list_datasets():
    if "parquet" in file and "int8" in file:
        if "training" in file or "validation" in file:
            napi.download_dataset(file, f"data/{file}")
        else:
            Path(f"data/{CURRENT_ROUND}").mkdir(exist_ok=True, parents=True)
            napi.download_dataset(file, f"data/{CURRENT_ROUND}/{file}")