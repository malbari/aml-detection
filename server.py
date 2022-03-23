import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import joblib as joblib
import flask
import numpy
import csv

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
from flask import request, jsonify
from logging.config import dictConfig
from numpy import array
from io import StringIO

random.seed(50)

pipeline4 = joblib.load('model.pkl')

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/predict', methods=['GET'])
def predict_value():
    missing = []

    if 'step' in request.args:
        step = int(request.args['step'])
    else:
        missing.append("step")

    if 'type' in request.args:
        atype = request.args['type']
    else:
        missing.append("type")

    if 'amount' in request.args:
        amount = float(request.args['amount'])
    else:
        missing.append("amount")

    if 'nameOrig' in request.args:
        nameOrig = request.args['nameOrig']
    else:
        missing.append("nameOrig")

    if 'oldbalanceOrg' in request.args:
        oldbalanceOrg = float(request.args['oldbalanceOrg'])
    else:
        missing.append("oldbalanceOrg")

    if 'newbalanceOrig' in request.args:
        newbalanceOrig = float(request.args['newbalanceOrig'])
    else:
        missing.append("newbalanceOrig")

    if 'nameDest' in request.args:
        nameDest = request.args['nameDest']
    else:
        missing.append("nameDest")

    if 'oldbalanceDest' in request.args:
        oldbalanceDest = float(request.args['oldbalanceDest'])
    else:
        missing.append("oldbalanceDest")

    if 'newbalanceDest' in request.args:
        newbalanceDest = float(request.args['newbalanceDest'])
    else:
        missing.append("newbalanceDest")

    if (len(missing) > 0):
       return jsonify(
            error="Missing one or more fields: " + str(missing)
        )

    #cols=["step","type","amount","nameOrig","oldbalanceOrg","newbalanceOrig","nameDest",#"oldbalanceDest","newbalanceDest","isFraud","isFlaggedFraud"]
    #rows=[[step,atype,amount,"_",oldbalanceOrg,newbalanceOrig,"_",oldbalanceDest,newbalanceDest,"1","1"]]
    
    header = "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
    row=str(step) +"," + atype + "," + str(amount) + "," + nameOrig + "," + str(oldbalanceOrg) + "," + str(newbalanceOrig) + "," + nameDest + "," + str(oldbalanceDest) + "," + str(newbalanceDest) + ",1,0"
    workaround="""
1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0
1,CASH_OUT,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0,1,0
1,PAYMENT,4024.36,C1265012928,2671.0,0.0,M1176932104,0.0,0.0,0,0
1,DEBIT,5337.77,C712410124,41720.0,36382.23,C195600860,41898.0,40348.79,0,0"""
    data=header+row+workaround
    #data=header+workaround

    app.logger.info("-------------------")
    app.logger.info("data is: " + data)
    app.logger.info("-------------------")
 
    #data=
    """
    step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
    1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0
    1,CASH_OUT,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0,1,0
    1,PAYMENT,4024.36,C1265012928,2671.0,0.0,M1176932104,0.0,0.0,0,0
    1,DEBIT,5337.77,C712410124,41720.0,36382.23,C195600860,41898.0,40348.79,0,0
    """
    dataset = pd.read_csv(StringIO(data))
    #dataset = pd.DataFrame(data=rows, columns=cols)
    #dataset = pd.read_csv('./predict-sample.csv')

    dataset.drop('nameOrig', axis=1, inplace=True)
    dataset.drop('nameDest', axis=1, inplace=True)
    dataset.drop('isFlaggedFraud', axis=1, inplace=True)
    X=dataset.iloc[:, :-1].values
    
    # Encoding categorical data
    labelencoder = LabelEncoder()
    X[:, 1] = labelencoder.fit_transform(X[:, 1])
    ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
    X = ct.fit_transform(X)

    # Feature Scaling
    sc = StandardScaler()
    X_val = sc.fit_transform(X)

    # predict output values
    y_val = pipeline4.predict(X_val)

    app.logger.info("y_val is: " + str(y_val))

    return jsonify (
        result=str(y_val[0])
    ) 

app.run()