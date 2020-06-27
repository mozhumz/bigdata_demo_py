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
user_feat=['gender','age','salary','province']
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
'''
W:
[[-2.59393937 -2.5153002  -1.1774175  -0.98807273 -0.90514931 -0.95982684
  -1.0787732  -0.96667477 -1.17050213 -1.04678741 -0.93045791 -0.99481736
   0.18343976  0.46643803 -0.38903157 -0.35600348 -0.24359633 -0.31009802
   0.35466513 -0.0636377  -0.60018864 -0.24072009  0.2157221  -0.24250333
  -0.35414788  0.43764482 -0.13976739 -0.43035663 -0.43247739  0.40837193
   0.0606094  -0.0860343  -0.04149778 -0.2052532  -0.30886219 -0.04942707
   0.33119144 -0.14021156 -0.14769334 -0.32180072 -0.01809694 -0.50387044
  -0.58370166 -0.22418789 -0.69891224 -0.4352444   0.28769899 -0.62802444
  -0.52034513  0.16064848 -0.43741211  0.62287386 -0.74439406 11.63195402]]
b:
[-5.10923957]
'''