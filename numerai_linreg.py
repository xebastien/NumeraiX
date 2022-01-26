# %% import packages
import numpy as np
import pandas as pd
import seaborn as sns           # statistical visualisation
import random


import matplotlib.pyplot as plt
%matplotlib inline
from scipy.stats import spearmanr
import gc

#!pip install numerapi
import numerapi
import matplotlib.pyplot as plt


# %% sklearn linear models
from sklearn import (
    feature_extraction, feature_selection, decomposition, linear_model,
    model_selection, metrics, svm
)
# %% load in memory
napi = numerapi.NumerAPI()

#Create instance of NumerAPI and open downloaed file
# need more memory
train_pq_path = "numerai_training_data_int8.parquet"
val_pq_path   = "numerai_validation_data_int8.parquet"


#napi.download_dataset("numerai_training_data_int8.parquet", train_pq_path)
#napi.download_dataset("numerai_validation_data_int8.parquet", val_pq_path)



#Read parquet files into DataFrames
df_train = pd.read_parquet('data/numerai_training_data_int8.parquet')  
df_val = pd.read_parquet('data/numerai_validation_data_int8.parquet') 
gc.collect()


# subsampling  as eras are overlaping
eras = [i for i in range(1, len(df_train.era.unique())+1, 4)]
df_train = df_train[df_train.era.astype(int).isin(eras)]
gc.collect()


# %% get features and target names

features = [c for c in df_train if c.startswith("feature")]
features_erano = features + ["erano"]

targets = [c for c in df_train if c.startswith("target")]
# cast era time from string to integer and replace in df
df_train["erano"] = df_train.era.astype(int)
TARGET = "target"   # new name since 2021

df_val["erano"] = df_val.era.astype(int)

# reindex 
df_train.index = pd.RangeIndex(start=0, stop=len(df_train), step=1)


# %% Metrics 
# 
# # The models should be scored based on the rank-correlation (spearman) with the target
def numerai_score(y_true, y_pred, eras):
    rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0,1]

# It can also be convenient while working to evaluate based on the regular (pearson) correlation
def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]

def spearman(y_true, y_pred): 
    return spearmanr(y_pred, y_true).correlation 


# %% LinearRegression/Lasso/Ridge whatever with partial data
# Reduce data again - keep only random features and see goodness of madel

# finding the most correlated(to scoring target) features across all eras 
all_feature_corrs = df_train.groupby("era").apply(lambda d: d[features].corrwith(d[TARGET]))
abs_corr = all_feature_corrs.mean().abs().sort_values(ascending=False)[:len(features)]
# usign only 30% of most correlated(to scoring target) features to reduce memory usage
# sel_features = abs_corr[:int(len(abs_corr)*0.5)].index
# or correlated variables are usually problematic in multivariate regressions - trying random selection
sel_features = random.sample(features, 30)

# plot correlation of randomly selected features
fig, ax = plt.subplots(figsize=(30, 15))
sns.heatmap(data=all_feature_corrs[sel_features], ax=ax)


# ridge regerssion
regr = linear_model.Ridge(alpha=0.1)
regr.fit(df_train[sel_features], df_train[TARGET])

print("Regr coeff  ")
print(str(regr.coef_)) 

print("Mean square error")
print(np.mean((regr.predict(df_train[sel_features]) - df_train[TARGET])**2))
print("Score")
regr.score(df_train[sel_features], df_train[TARGET])





# %% ------------------------------------------------------------------------
# KBoost with eras from past to future

from TimeSeriesSplitGroups import TimeSeriesSplitGroups

# store score erawise
scores = []
    
y_df =  df_train["target"].to_frame()  # keep frame
x_df = df_train[features + ["erano"]]
# K_split = 10
tSSG = TimeSeriesSplitGroups(n_splits=5) 
# split as TimeSeries    
for i,(train,test) in enumerate(tSSG.split(X=x_df, y=y_df, groups=df_train.erano)):
    #lgbm_model = lgb.DaskLGBMRegressor()
    # lgbm_model.fit(X_arr[train], y_arr[train])
        
    regr = linear_model.Ridge(alpha=0.1)
    regr.fit(x_df.loc[train], y_df.loc[train])
    # predict on subsequent data
    predictions = regr.predict(x_df.loc[test])

    scores.append(spearman(y_df.loc[test], predictions))
        
        
print("Scores     "  + str(scores))

print("Mean score   " + str(np.mean(scores)))






# %%
