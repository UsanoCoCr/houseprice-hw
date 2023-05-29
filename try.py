import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import ElasticNet

from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score

from scipy import stats

train = pd.read_csv("./data/train.csv")
train_y = train[["SALE PRICE"]]
train_X = train
del train_X["SALE PRICE"]

test_X = pd.read_csv("./data/test.csv")
test_y = pd.read_csv("./data/test_groundtruth.csv")

num_train_samples = len(train_X)
data_X = pd.concat([train_X, test_X])

del data_X['ADDRESS']
del data_X['APARTMENT NUMBER']
""" del data_X['BUILDING CLASS AT PRESENT']
del data_X['BUILDING CLASS AT TIME OF SALE']
del data_X['NEIGHBORHOOD']
del data_X['SALE DATE']
del data_X['LAND SQUARE FEET']
del data_X['GROSS SQUARE FEET'] """

data_X['SALE DATE'] = pd.to_datetime(data_X['SALE DATE']).astype(np.int64)

def convert_to_number(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

data_X['LAND SQUARE FEET'] = data_X['LAND SQUARE FEET'].map(convert_to_number)
data_X['GROSS SQUARE FEET'] = data_X['GROSS SQUARE FEET'].map(convert_to_number)
data_X['YEAR BUILT'] = data_X['YEAR BUILT'].map(convert_to_number)

data_X['LAND SQUARE FEET'].replace(0, np.nan, inplace=True)
data_X['GROSS SQUARE FEET'].replace(0, np.nan, inplace=True)
data_X['YEAR BUILT'].replace(0, np.nan, inplace=True)

# 按NEIGHBORHOOD的中位数填充，如果某个NEIGHBORHOOD的所有LAND SQUARE FEET、GROSS SQUARE FEET或YEAR BUILT数据都是缺失的，则使用全局中位数填充
data_X['LAND SQUARE FEET'] = data_X.groupby('NEIGHBORHOOD')['LAND SQUARE FEET'].transform(
    lambda x: x.fillna(x[x>0].median()) if x[x>0].median() >= 0 
                                        else data_X['LAND SQUARE FEET'][data_X['LAND SQUARE FEET']>0].median())
data_X['GROSS SQUARE FEET'] = data_X.groupby('NEIGHBORHOOD')['GROSS SQUARE FEET'].transform(
    lambda x: x.fillna(x[x>0].median()) if x[x>0].median() >= 0 
                                        else data_X['GROSS SQUARE FEET'][data_X['GROSS SQUARE FEET']>0].median())
data_X['YEAR BUILT'] = data_X.groupby('NEIGHBORHOOD')['YEAR BUILT'].transform(
    lambda x: x.fillna(x[x>0].median()) if x[x>0].median() >= 0 
                                        else data_X['YEAR BUILT'][data_X['YEAR BUILT']>0].median())

# 数据类型转换
data_X['LAND SQUARE FEET'] = data_X['LAND SQUARE FEET'].astype(np.int64)
data_X['GROSS SQUARE FEET'] = data_X['GROSS SQUARE FEET'].astype(np.int64)
data_X['YEAR BUILT'] = data_X['YEAR BUILT'].astype(np.int64)

data_X['TAX CLASS AT TIME OF SALE'] = data_X['TAX CLASS AT TIME OF SALE'].astype('category')
data_X['TAX CLASS AT PRESENT'] = data_X['TAX CLASS AT PRESENT'].astype('category')
data_X['BOROUGH'] = data_X['BOROUGH'].astype('category')
#data_X['ADDRESS'] = data_X['ADDRESS'].astype('category')
#data_X['APARTMENT NUMBER'] = data_X['APARTMENT NUMBER'].astype('category')
data_X['BUILDING CLASS AT PRESENT'] = data_X['BUILDING CLASS AT PRESENT'].astype('category')
data_X['BUILDING CLASS AT TIME OF SALE'] = data_X['BUILDING CLASS AT TIME OF SALE'].astype('category')
data_X['NEIGHBORHOOD'] = data_X['NEIGHBORHOOD'].astype('category')

one_hot_features = ['BOROUGH', 'BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE',
                    'BUILDING CLASS AT PRESENT','BUILDING CLASS AT TIME OF SALE','NEIGHBORHOOD']

one_hot_encoded = pd.get_dummies(data_X[one_hot_features])

data_X = data_X.drop(one_hot_features,axis=1)
data_X = pd.concat([data_X, one_hot_encoded] ,axis=1)

train_X = data_X[:num_train_samples].to_numpy()
test_X = data_X[num_train_samples:].to_numpy()

rf_regr = RandomForestRegressor()
rf_regr.fit(train_X, train_y)
Y_pred_rf = rf_regr.predict(test_X)

# MAPE metric
print(mean_absolute_percentage_error(test_y,Y_pred_rf))