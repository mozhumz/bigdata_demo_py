import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square

IDIR = 'D://data//data//'
df_train = pd.read_csv(IDIR + 'train_feat.csv').fillna(0.).to_sparse()
print(df_train)
labels = np.load(IDIR + 'labels.npy')
print(labels)

X_train, X_test, y_train, y_test = train_test_split(df_train, labels, test_size=0.2, random_state=2019)
del df_train
del labels
# [买：100，不买：10]

# dt = DecisionTreeClassifier()

rfr = RandomForestRegressor(n_estimators=10,
                            criterion="mse",
                            max_depth=None,
                            min_samples_split=200,
                            min_samples_leaf=100,
                            min_weight_fraction_leaf=0.,
                            max_features="auto",
                            max_leaf_nodes=None,
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=3,
                            random_state=None,
                            verbose=0,
                            warm_start=False)

rfr.fit(X_train, y_train)
print(rfr.feature_importances_)

y_pred = rfr.predict(X_test)
mse_test = np.sum((y_pred - y_test) ** 2) / len(y_test)
print('mse : ', mse_test)

rmse_test = mse_test ** 0.5
print('Rmse : ', rmse_test)

print('mean_absolute_error: ', mean_absolute_error(y_test, y_pred))
print('mean_squared_error: ', mean_squared_error(y_test, y_pred))
print('r2_score: ', r2_score(y_test, y_pred))
