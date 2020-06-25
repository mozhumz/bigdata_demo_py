import demoDay21_recsys_music.hyj.gen_cf_data_hyj as gen
import demoDay21_recsys_music.hyj.config_hyj as conf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows',None)

import numpy as np
np.set_printoptions(threshold = np.inf)
#若想不以科学计数显示:
np.set_printoptions(suppress = True)


user_item_df=gen.user_item_socre(5000)

music_data=conf.music_data()
user_profile=conf.user_profile()




# pd.merge(user_item_df,user_profile,how='inner',on='user_id')

data=user_item_df.merge(music_data,how='inner',on='item_id').merge(user_profile,how='inner',on='user_id')
# 指定每条数据的y值
def score2Label(score):
    if score>0.9:
        return 1
    return 0
labels=data['label']=data['score'].apply(score2Label)

print(labels)
'''
离散特征 one-hot处理
'''
user_feat=['user_id','gender','age','salary','province']
discrete_feat=user_feat+['location']
continue_feat=['score']

df=pd.get_dummies(data[discrete_feat])
'''
 连续特征score添加到上述df
'''
df[continue_feat]=data[continue_feat]

print(df.head(20))

'''
 划分训练集和测试集
'''
X_train,X_test,y_train,y_test=train_test_split(df,labels,test_size=0.3,random_state=2020)

'''
LR模型训练 fit
'''
lr=LogisticRegression(penalty='l2',dual=False,tol=1e-4,C=1.0,fit_intercept=True,intercept_scaling=1,class_weight=None,
                      random_state=None,solver='liblinear',max_iter=100,multi_class='ovr',verbose=1,warm_start=False,n_jobs=-1)


model=lr.fit(X_train,y_train)

W=lr.coef_
b=lr.intercept_
print('W:')
print(W)

print("b:")
print(b)