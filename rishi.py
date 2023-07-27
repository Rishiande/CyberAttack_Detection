import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import regularizers
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read Train and Test dataset
data_train = pd.read_csv("/content/KDDTrain+.txt")

columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'])

# Assign name for columns
data_train.columns = columns

#Remplacement les valeurs dans la colonne 'outcome'
data_train.loc[data_train['outcome'] == "normal", "outcome"] = 'normal'
data_train.loc[data_train['outcome'] != 'normal', "outcome"] = 'attack'

# Fonction pour effectuer une mise à l'échelle des données numériques
def Scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=cols)
    return std_df

cat_cols = ['is_host_login','protocol_type','service','flag','land', 'logged_in','is_guest_login', 'level', 'outcome']
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

# Prétraitement des données d'entraînement
scaled_train = preprocess(data_train)

x = scaled_train.drop(['outcome', 'level'], axis=1).values
y = scaled_train['outcome'].values
y_reg = scaled_train['level'].values

# Sélection des variables prédictives en excluant les colonnes 'outcome' et 'level' du DataFrame "scaled_train"
x = scaled_train.drop(['outcome', 'level'], axis=1).values

# Extraction de la variable cible 'outcome' dans "y"
y = scaled_train['outcome'].values

# Extraction de la variable cible de régression 'level' dans "y_reg"
y_reg = scaled_train['level'].values

# Réduction de dimension avec PCA
pca = PCA(n_components=20)
pca = pca.fit(x)
x_reduced = pca.transform(x)

# Conversion du type de la variable cible "y" en entier
y = y.astype('int')

# Séparation des données en ensembles d'entraînement et de test pour la classification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
xgb_classifier.fit(x_train, y_train)

# Function to evaluate classification model
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))

    train_precision = metrics.precision_score(y_train, model.predict(X_train))
    test_precision = metrics.precision_score(y_test, model.predict(X_test))

    train_recall = metrics.recall_score(y_train, model.predict(X_train))
    test_recall = metrics.recall_score(y_test, model.predict(X_test))

    print("Model: ", name)
    print("Training Accuracy: {:.2f}%  Test Accuracy: {:.2f}%".format(train_accuracy*100, test_accuracy*100))
    print("Training Precision: {:.2f}%  Test Precision: {:.2f}%".format(train_precision*100, test_precision*100))
    print("Training Recall: {:.2f}%  Test Recall: {:.2f}%".format(train_recall*100, test_recall*100))

    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['normal', 'attack'])

    fig, ax = plt.subplots(figsize=(10,10))
    ax.grid(False)
    cm_display.plot(ax=ax)

# Evaluate the XGBoost classifier
evaluate_classification(xgb_classifier, "XGBoost", x_train, x_test, y_train, y_test)

# Streamlit App
def main():
    st.title("Intrusion Detection System using XGBoost")
    st.write("This is a web app to detect network intrusions using the XGBoost classifier.")

    # Add your Streamlit app code here

if __name__ == "__main__":
    main()
