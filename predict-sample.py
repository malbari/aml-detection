# Inspired by https://www.kaggle.com/x09072993/aml-detection
# 
# Use model "model.plk" to predict sample file "predict-sample.csv"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import joblib as joblib

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from imblearn.under_sampling import NearMiss
from imblearn import over_sampling as os
from imblearn.pipeline import make_pipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix, precision_score,auc,roc_auc_score,roc_curve,recall_score

random.seed(50)

# Importing the dataset
dataset = pd.read_csv('./predict-sample.csv')
dataset.drop('nameOrig', axis=1, inplace=True)
dataset.drop('nameDest', axis=1, inplace=True)
dataset.drop('isFlaggedFraud', axis=1, inplace=True)
X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 7].values

print("X is: " + str(X))
print("X type is: " + str(type(X)))

# Encoding categorical data
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Feature Scaling
sc = StandardScaler()
X_val = sc.fit_transform(X)

# Load the pipeline from "model.pkl"
pipeline4 = joblib.load('model.pkl')

# predict output values
y_val = pipeline4.predict(X_val)

print("y_val is: " + str(y_val))

# print out values
#i=0
#while i<len(y_val):
#    print("[" + str(i) + "]: " + str(y_val[i]))
#    i=i+1

print("done")