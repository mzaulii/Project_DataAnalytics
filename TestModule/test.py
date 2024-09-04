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
        X = df.drop(columns=['Year']) 
        y = df['Year'] 
        scaler = pickle.load(open("scaler.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        dfNew = pd.concat([X, y], axis = 1)
        return dfNew
    else: # No scaler per TabNet e TabTransformer
        return df


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
    X = df.drop(columns=['Year']) 
    y = df['Year'] 
    
    #ypred = clf.predict(X)

    # Tabular
    if (clfName == "TB") or (clfName == "TF"):
        ypred = clf.predict(df)  

    # Rete FF
    elif clfName == "FF":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.FloatTensor(X.values).to(device)
        y = torch.FloatTensor(y.values).view(-1, 1).to(device)
        clf.eval()

        with torch.no_grad():  
            ypred_ = clf(X)  # Ottengo le predizioni
            ypred = ypred_.cpu().numpy()  # Converto le predizioni da tensor a numpy array per calcolare le metriche
            
        y = y.cpu().numpy()  # Converto y in numpy array per calcolare le metriche

    else:
        X = X.values  
        y = y.values  
        ypred = clf.predict(X)  


    mse = mean_squared_error(y, ypred)
    mae = mean_absolute_error(y, ypred)
    mape = mean_absolute_percentage_error(y, ypred)
    r2 = r2_score(y, ypred)
    
    perf = {
        "mse": mse, 
        "mae": mae, 
        "mape": mape, 
        "r2square": r2
    }
    
    return perf
    

