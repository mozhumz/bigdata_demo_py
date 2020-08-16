import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib


IDIR = 'G:\\bigdata\\badou\\00-data//'
df_train = pd.read_csv(IDIR + 'train_feat.csv').fillna(0.).astype(pd.SparseDtype("float", np.nan))
labels = np.load(IDIR + 'labels.npy')
X_train, X_test, y_train, y_test = train_test_split(df_train, labels, test_size=0.2, random_state=2020)
print('load_model')
rfr=joblib.load('randomForestRegressor.m')
print('W:',rfr.feature_importances_)
y_pred = rfr.predict(X_test)
# y_pred_train = rfr.predict(X_train)
# 0.8272266140627522
print('auc_test0:',roc_auc_score(y_test,y_pred))
# print('auc_train0:',roc_auc_score(y_train,y_pred_train))
print('train again...')
# 模型再训练
rfr.fit(X_train, y_train)
print('W2:',rfr.feature_importances_)
y_pred = rfr.predict(X_test)
print('auc_test1:',roc_auc_score(y_test,y_pred))