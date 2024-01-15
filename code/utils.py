from typing import Tuple, Union, List
import openml
import csv
import os
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from flwr.common import NDArrays
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import datetime
import warnings


# XY = Tuple[np.ndarray, np.ndarray]
# Dataset = Tuple[XY, XY]
# LogRegParams = Union[XY, Tuple[np.ndarray]]
# XYList = List[XY] 


def get_model_parameters(model: LogisticRegression) -> NDArrays:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = 4  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def load_sleep_data():
    # Read the CSV file into a Pandas DataFrame
    df_client4=pd.read_csv('C:\\Users\\smarthala\\OneDrive - Microsoft\\Desktop\\sleep\\dataset\\client4_dataset.csv')
    df_client3=pd.read_csv('C:\\Users\\smarthala\\OneDrive - Microsoft\\Desktop\\sleep\\dataset\\client3_dataset.csv')
    df_client2=pd.read_csv('C:\\Users\\smarthala\\OneDrive - Microsoft\\Desktop\\sleep\\dataset\\client2_dataset.csv')
    df_client1=pd.read_csv('C:\\Users\\smarthala\\OneDrive - Microsoft\\Desktop\\sleep\\dataset\\client1_dataset.csv')

    # Feature Engineering

    # I merged four files into one
    df=pd.concat([df_client1, df_client2,df_client3,df_client4])

    # There are some NaNs in data. So I droped NaN.
    df=df.dropna()

    # I found some different type of data in some coulumns, so I fixed them.
    df['HOURS OF SLEEP'] = df['HOURS OF SLEEP'].replace('6:46', '6:46:00')
    df['HOURS OF SLEEP'] = df['HOURS OF SLEEP'].replace('6:06', '6:06:00')

    # I changed 'DATE' to Datetime format and,
    # Changed 'HOURS OF SLEEP' from timedelta64 to int, 'second'.
    # I also made 'Month', 'Week' and 'Day' columns.

    df['DATE']=pd.to_datetime(df['DATE'],format='%m/%d/%Y')

    baseline=pd.to_datetime('00:00:00',format='%H:%M:%S')
    df['HOURS OF SLEEP']=pd.to_datetime(df['HOURS OF SLEEP'],format='%H:%M:%S')-baseline
    df['SECONDS OF SLEEP'] = df['HOURS OF SLEEP'].astype('int64') // 1000000000
                                        
    df['Week']=df['DATE'].dt.weekday
    df['Month']=df['DATE'].dt.month
    df['Day']=df['DATE'].dt.day

    # I changed 'REM SLEEP', 'DEEP SLEEP' and 'HEART RATE BELOW RESTING' to float.
    df['REM SLEEP']=df['REM SLEEP'].str[:-1]
    df['DEEP SLEEP']=df['DEEP SLEEP'].str[:-1]
    df['HEART RATE BELOW RESTING']=df['HEART RATE BELOW RESTING'].str[:-1]

    df['REM SLEEP']=df['REM SLEEP'].astype(float)/100
    df['DEEP SLEEP']=df['DEEP SLEEP'].astype(float)/100
    df['HEART RATE BELOW RESTING']=df['HEART RATE BELOW RESTING'].astype(float)/100

    # I split 'SLEEP TIME' to 'Sleep_start' and 'Sleep_end' columns.
    df['SLEEP TIME'] = df['SLEEP TIME'].replace('11:21 - 8:45am', '11:21pm - 8:45am')
    df['SLEEP TIME'] = df['SLEEP TIME'].replace('11:40pm - 7:33', '11:40pm - 7:33am')
    df['SLEEP TIME'] = df['SLEEP TIME'].replace('11:16pm - 7:02', '11:16pm - 7:02am')
    df['SLEEP TIME'] = df['SLEEP TIME'].replace('11-38pm - 8:23am', '11:38pm - 8:23am')

    df1=df['SLEEP TIME'].str.split('-', expand=True)
    df1.columns = ['Sleep_start', 'Sleep_end']

    df1['Sleep_start']=df1['Sleep_start'].str[:-3]
    df1['Sleep_end']=df1['Sleep_end'].str[:-2]
    df1['Sleep_end']=df1['Sleep_end'].str[0:]

    df1['Sleep_end'] = df1['Sleep_end'].str.replace(' ', '')

    df1['Sleep_start']=pd.to_datetime(df1['Sleep_start'],format='%H:%M')
    df1['Sleep_end']=pd.to_datetime(df1['Sleep_end'],format='%H:%M')

    df=pd.concat([df, df1],axis=1)

    df['Sleep_start']=df['Sleep_start'].dt.time
    df['Sleep_end']=df['Sleep_end'].dt.time

    df=df.drop(['SLEEP TIME','HOURS OF SLEEP'],axis=1)

    df = df.reset_index()
    df=df.drop('index',axis=1)

    df1=df.drop(['Day','Week','Month'],axis=1)

    # I split data to 'over 80 SLEEP SCORE' =1 and 'below 80 SLEEP SCORE'=0 to classify data good or bad.
    def score_judge(ex):
        if ex >= 80:
            return 1
        else:
            return 0

    if 'SLEEP SCORE' in df1.columns:
        df1.loc[:, 'Evaluation'] = df1.loc[:, 'SLEEP SCORE'].apply(score_judge)
    else:
        print("Column 'SLEEP SCORE' not found in DataFrame.")


    # I can find the difference between evaluation 0 and 1 in 'REM SLEEP' ,'DEEP SLEEP','SECONDS OF SLEEP' and 'HEART RATE BELOW RESTING'

    df2=df1.drop(['DATE', 'DAY','Sleep_start','Sleep_end','SLEEP SCORE'],axis=1)


    X=df2.drop('Evaluation',axis=1).values
    y=df2['Evaluation']
    X_norm = (X - np.min(X)) / (np.max(X))

    X_train, X_test, y_train, y_test = train_test_split(X_norm,y,test_size=0.8,random_state=42)

    # Return the preprocessed data
    return (X_train, y_train), (X_test, y_test)


