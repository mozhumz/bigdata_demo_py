import xgboost as xgb
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split


# pip install xgboost,lightgbm,sklearn  anaconda3
IDIR = 'G://bigdata//badou//00-data//'
df_train = pd.read_csv(IDIR + 'train_feat.csv').fillna(0.).to_sparse()
labels = np.load(IDIR + 'labels.npy')

X_train, X_test, y_train, y_test = train_test_split(df_train, labels,
                                                    test_size=0.2,
                                                    random_state=2020)
del df_train
del labels

# ########################### XGB ##################

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_test, y_test)

param = {'booster': 'gbtree',
         'gamma': 0.1,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'max_depth': 6,
         'eta': 0.03,
         'silent': 1,
         'objective': 'binary:logistic',
         'nthread': 4,
         'eval_metric': 'auc'}
watchlist = [(dtrain, 'train'), (dval, 'val')]
# pip install xgboost
xgb_start = datetime.now()
model = xgb.train(param, dtrain, num_boost_round=100, evals=watchlist)
xgb_stop = datetime.now()
execution_time_lgbm = xgb_stop - xgb_start
print('xgb time cost: ', execution_time_lgbm / 1000)
# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)
#
# 计算准确率
cnt1 = 0.
cnt2 = 0.
for i in range(len(y_test)):
    if ans[i] > 0.3:
        cnt1 += 1.
    else:
        cnt2 += 1.

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))


# ##################lightGBM#########################
params_lgbm = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'binary',
    # 'objective': 'regression', # 目标函数     ####regression默认regression_l2
    # 'metric': 'rmse',  # 评估函数
    'metric': 'auc',
    'max_depth': 6,  #   树的深度           ###按层
    'num_leaves': 50,  #   由于leaves_wise生长，小于2^max_depth   #####按leaf_wise
    'learning_rate': 0.05,  # 学习速率,步长
    'subsample': 0.8,  #  数据采样
    'colsample_bytree': 0.8,  #  特征采样
}

train_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_test, y_test)

num_round = 100
start = datetime.now()
lgbm = lgb.train(params_lgbm, train_data,
                 num_round,
                 valid_sets=[train_data, val_data],
                 early_stopping_rounds=50)
stop = datetime.now()
# lightgbm
execution_time_lgbm = stop - start
print('lightgbm time cost: ', execution_time_lgbm)
