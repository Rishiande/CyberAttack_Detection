import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import regularizers
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Function for preprocessing
def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = Scaling(df_num, num_cols)

    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]

    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1

    dataframe = pd.get_dummies(dataframe, columns=['protocol_type', 'service', 'flag'])
    return dataframe


# Function for evaluation
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    # Your evaluation code here...
    pass

def main():
    st.title('Machine Learning Model Deployment')
    st.write('Load your data here and create your Streamlit app.')

    # Load your data and preprocess it
    data_train = pd.read_csv("/content/KDDTrain+.txt")
    columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
    ,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
    ,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
    ,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
    ,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
    ,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'])

    data_train.columns = columns

    data_train.loc[data_train['outcome'] == "normal", "outcome"] = 'normal'
    data_train.loc[data_train['outcome'] != 'normal', "outcome"] = 'attack'

    scaled_train = preprocess(data_train)

    x = scaled_train.drop(['outcome', 'level'], axis=1).values
    y = scaled_train['outcome'].values

    # Train the XGBoost classifier
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
    xgb_classifier.fit(x_train, y_train)

    # Evaluate the XGBoost classifier
    evaluate_classification(xgb_classifier, "XGBoost", x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()
