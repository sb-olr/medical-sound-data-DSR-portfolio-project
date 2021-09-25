#Data visualization

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Audio Analysis
import glob
import IPython
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras

#path
import os

csv_name = './CoronaHack-Respiratory-Sound-Dataset/Corona-Hack-Respiratory-Sound-Metadata.csv'
base_path = './CoronaHack-Respiratory-Sound-Dataset'

def open_csv(csv_name):
    # import meta data
    # Meta data csv contain different additional information about each case.
    # csv file contains the path to the .wav files for each feature and each case
    df_meta = pd.read_csv(csv_name)
    print(df_meta.info(), df_meta.shape)
    return df_meta

def get_label(df_meta):
    #Get the label (healthy / COVID) 

    #split COVID STATUS column to get labels in column 'split'
    df_meta['split'] = df_meta['COVID_STATUS'].str.split('_').str.get(0)
    #Check for NA
    df_meta.loc[:,'counting-normal'].isna().sum()
    df_meta.loc[:,'split'].value_counts()

    #Generate a dict to re-categorize the split column
    cat_dict = {'healthy':0,'no':0,'resp':0,'recovered':0,'positive':1}

    #map cat_dict to split column 
    df_meta.loc[:,'split'] =  df_meta.loc[:,'split'].map(cat_dict)
    df_meta2 = df_meta.dropna(subset=['split'])
    df_meta2.loc[:,'split'] = df_meta2.loc[:,'split'].astype('int32')


    #Extract positive USER ID
    df_meta_positives = df_meta[df_meta['split'] == 1]
    df_meta_negatives = df_meta[df_meta['split'] == 0]

    positives = list(df_meta_positives['USER_ID'])
    negatives = list(df_meta_negatives['USER_ID'])
    len(positives),len(negatives)
    #positives
    return df_meta2


def get_paths(df, col_name):
    column = df[col_name].tolist()
    return column





df_meta = open_csv(csv_name)

#print(df_meta.head()) 

#print(get_paths(df_meta,'counting-normal'))

df_meta2 = get_label(df_meta)
print(df_meta2.head()) 

