
###########################################################################
#  Neural network training for numer.ai
#
##########################################################################
# %% import package
import tensorflow as tf                         # call gpu tf if available (CUDA required)
from tensorflow.keras import layers, models     # keras tf v2
import numpy as np
import matplotlib
import pandas as pd
import time
#import tensorflow_probability as tfp
# from keras.callbacks import EarlyStopping     # need to insert callbacks

import gc                                       # garbage collector / needed for my laptop
#import lightgbm as lgb
import matplotlib.pyplot as plt

#from scipy.stats import spearmanr

# will need to use dask for cluster
# import dask.dataframe as dd # work on external clusters
# from dask.array import from_array

# look for S3 bucket below for loading in cloud
# public S3

# %%Create instance of NumerAPI and open downloaed file
if 0:
    import numerapi     # numerai api 
    napi = numerapi.NumerAPI(verbosity="info")
    # download current dataset
    napi.download_current_dataset(unzip=True)

    # getting the latest round information
    current_ds = napi.get_current_round()
    # latest_round = os.path.join('numerai_dataset_'+str(current_ds))

    ## already downloaded
    #napi.download_dataset("numerai_training_data_int8.parquet", train_pq_path)
    #napi.download_dataset("numerai_validation_data_int8.parquet", val_pq_path)

#  memory - using  parquet/int8 data file for now
train_pq_path = "numerai_training_data_int8.parquet"
val_pq_path   = "numerai_validation_data_int8.parquet"


#Read parquet files and put to DataFrames
df_train = pd.read_parquet('Numerai/data/numerai_training_data_int8.parquet')  
df_val   = pd.read_parquet('Numerai/data/numerai_validation_data_int8.parquet') 

# %% Features names and eras
features = [c for c in df_train if c.startswith("feature")]
features_erano = features + ["erano"]

targets = [c for c in df_train if c.startswith("target")]
# not used here, times series disabled
# cast era time from string to integer and store in df
df_train["erano"] = df_train.era.astype(int)
eras = df_train.erano
df_val["erano"] = df_val.era.astype(int)

print(f"Loaded {len(features)} features colum names")

# %% Create tf tensors
gc.collect()

x_train = df_train.reset_index()[features].to_numpy()
y_train = df_train.reset_index()["target"].to_numpy()
# time series
# x_train_erano = df_train.reset_index()[features_erano].to_numpy()

del df_train; gc.collect()  # low on memory
print("Tensor training data ok - df dropped")
x_test = df_val.reset_index()[features].to_numpy()
y_test = df_val.reset_index()["target"].to_numpy()

del df_val; gc.collect()     # low on memory
print("Tensor validation data ok - df dropped")
# slicing data for batch processing 
batch_size = len(x_test) // 100
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
print("Tensor sliced")
# %% Define model - optimizer - loss function
epochs = 15
# model here
#leaky_relu = LeakyReLU(alpha=0.01)

model = models.Sequential([
        layers.Dense(1000, activation='relu', kernel_initializer='normal',input_shape=[len(features)]),
        layers.Dense(50, activation='elu', kernel_regularizer='l2'),
        layers.Dense(16, activation='relu', kernel_regularizer='l2'),
        layers.Dense(1)  # classify in [0 0.25 0.5 0.75 1]
])
# Adam ok
optimizer   = tf.keras.optimizers.Adam()
# define loss objecctives
loss_object = tf.keras.losses.MeanSquaredError()

# or custum correlation lost funtion for regression 
# def MaxCorrelation(y_true,y_pred):
#    return -tf.math.abs(tfp.stats.correlation(y_pred,y_true, sample_axis=None, event_axis=None))
#loss_object = MaxCorrelation()

# metrics 
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.MeanSquaredError()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.MeanSquaredError()


## %% define functions 
@tf.function
def train_step(train_ds, labels):
    with tf.GradientTape() as tape:
        predictions=model(train_ds)
        loss = loss_object(labels, predictions)
        #loss = MaxCorrelation(y_true,y_pred)
    gradients=tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)                # averaging loss
    train_accuracy(labels, predictions)

def train(X_train, epochs):
    for itrain in range(epochs):
        start=time.time()
        # train by batch
        for train_ds, labels in X_train:
            train_step(train_ds, labels)
        # verbose
        message='Epoch {:04d}, loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
        print(message.format(itrain+1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            time.time()-start))
        train_loss.reset_states()
        train_accuracy.reset_states()

def test(test_ds):
    start=time.time()
    for test_x, test_labels in test_ds:
        predictions = model(test_x)
        t_loss=loss_object(test_labels, predictions)
        test_loss(t_loss)       # averaging
        test_accuracy(test_labels, predictions)
    message='Loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
    print(message.format(test_loss.result(),
                        test_accuracy.result()*100,
                        time.time()-start))




# %%  Run optimization and prediction on validation 
print("Training dataset - Optimization")
train(train_ds, epochs)
print("Validation dataset")
test(test_ds)

y_pred = model(x_test).numpy().reshape((-1,))
y_true = y_test
# %% metrics



# Score based on the rank-correlation (spearman) / eras
def numerai_score(y_true, y_pred, eras):
    rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0,1]

# Pearson correlation
def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]

# rank correlation  no era
from scipy.stats import spearmanr
def spearman(y_true, y_pred): 
    return spearmanr(y_pred, y_true).correlation 

# sum of square mean difference
def ssmd(y_true, y_pred):
   squared_difference = tf.square(y_true - y_pred)
   return tf.reduce_mean(squared_difference, axis=-1)


# remove warnings
tf.autograph.set_verbosity(0)

# %% ###############################################################################
# upload prediction
# import numerapi
# napi = numerapi.NumerAPI("xebastien", "")

# download data
# napi.download_current_dataset(unzip=True)




# upload predictions
# napi.upload_predictions("predictions.csv", model_id="model_id")
# %%
