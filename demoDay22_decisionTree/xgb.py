import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

IDIR = 'D://data//data//'
df_train = pd.read_csv(IDIR + 'train_feat.csv').fillna(0.).to_sparse()
labels = np.load(IDIR + 'labels.npy')

X_train, X_test, y_train, y_test = train_test_split(df_train, labels,
                                                    test_size=0.2,
                                                    random_state=2019)
del df_train
del labels

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_test,y_test)
param = {'booster': 'gbtree',
         'gamma': 0.1,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'max_depth': 6,
         'eta': 0.03,
         'silent': 0,
         'objective': 'binary:logistic',
         'nthread': 4,
         'eval_metric': 'auc'}
watchlist = [(dtrain,'train'),(dval,'val')]
# pip install xgboost
model = xgb.train(param,dtrain,100,evals=watchlist)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

# 计算准确率
cnt1 = 0.
cnt2 = 0.
for i in range(len(y_test)):
    if ans[i] > 0.3:
        cnt1 += 1.
    else:
        cnt2 += 1.

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
