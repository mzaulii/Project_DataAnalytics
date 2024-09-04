#from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import pickle
import pandas as pd
import torch


MY_UNIQUE_ID = "martinaallaseconda"


# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID

# Input: Test dataframe
# Output: PreProcessed test dataframe
def preprocess(df, clfName):

    # DOBIAMO METTERE ANCHE TABULAR???
    if ((clfName == "RF") or (clfName == "LR") or (clfName == "SVR") or (clfName == "KNR") or (clfName == "FF")):
        X = df.iloc[:, :5] # ?????
        y = df.iloc[:, 5] # ?????
        scaler = pickle.load(open("scaler.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        dfNew = pd.concat([X, y], axis = 1)
        return dfNew



# Input: Regressor name ("lr": Linear Regression, "SVR": Support Vector Regressor)
# Output: Regressor object
def load(clfName):
    if (clfName == "RF"):
        clf = pickle.load(open("rf.save", 'rb'))
        return clf
    elif (clfName == "LR"):
        clf = pickle.load(open("lr.save", 'rb'))
        return clf
    elif (clfName == "SVR"):
        clf = pickle.load(open("svr.save", 'rb'))
        return clf
    elif (clfName == "KNR"):
        clf = pickle.load(open("knn.save", 'rb'))
        return clf
    elif (clfName == "FF"):
        clf = pickle.load(open("ff4.save", 'rb'))
        return clf
    elif (clfName == "TB"):
        clf = pickle.load(open("tb_tabnet4.save", 'rb'))
        return clf
    elif (clfName == "TF"):
        clf = pickle.load(open("tabTransf_best.save", 'rb'))
        return clf
    else:
        return None



# Input: PreProcessed dataset, Regressor Name, Regressor Object 
# Output: Performance dictionary
def predict(df, clfName, clf):
    X = df.iloc[:, :5] # ?????
    y = df.iloc[:, 5] # ?????
    ypred = clf.predict(X)
    mse = mean_squared_error(ypred, y)
    mae = mean_absolute_error(ypred, y)
    mape = mean_absolute_percentage_error(ypred, y)
    r2 = r2_score(ypred, y)
    perf = {"mse": mse, "mae": mae, "mape": mape, "r2square": r2}
    return perf
    
    
    
    
    
    