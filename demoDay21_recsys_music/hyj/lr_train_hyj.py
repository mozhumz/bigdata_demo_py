import demoDay21_recsys_music.hyj.config_hyj as conf
import pandas as pd
import demoDay21_recsys_music.hyj.gen_cf_data_hyj as gen
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import common.common_util as util

data=gen.user_item_socre(nrows=50000)

# 定义label stay_seconds/total_timelen>0.9 -> 1
data['label']=data['score'].apply(lambda x:1 if x>=0.9 else 0)
# 关联用户信息和item信息到data
user_profile=conf.user_profile()
music_meta=conf.music_data()

# data数据结构
# user_id	                            item_id	score	label	gender	age	salary	province	item_name	total_timelen	location	tags
# 0	0000066b1be6f28ad5e40d47b8d3e51c	426100349	1.280	1	女	26-35	10000-20000	香港	刘德华 - 回家的路 2015央视春晚 现场版	250	港台	-
# 1	000072fc29132acaf20168c589269e1c	426100349	1.276	1	女	36-45	5000-10000	湖北	刘德华 - 回家的路 2015央视春晚 现场版	250	港台	-
# 2	000074ec4874ab7d99d543d0ce419118	426100349	1.084	1	男	36-45	2000-5000	宁夏	刘德华 - 回家的路 2015央视春晚 现场版	250	港台	-
data=data.merge(user_profile,how='inner',on='user_id').merge(music_meta,how='inner',on='item_id')

''' 定义特征X'''

#用户特征
user_feat=['age','gender','salary','province']
# 物品特征
item_feat=['location','total_timelen']
item_text_feat=['item_name','tags']
# 交叉特征
watch_feat=['stay_seconds','score','hour']

# 离散和连续特征
dispersed_feat=user_feat+['location']
continue_feat=['score']

# 获取Y （label）
labels=data['label']
del data['label']

# 离散特征-one-hot处理
# df数据结构 ：
# age_0-18	age_19-25	age_26-35	age_36-45	age_46-100	gender_女	gender_男	salary_0-2000	salary_10000-20000	salary_2000-5000	...	province_香港	province_黑龙江	location_-	location_亚洲	location_国内	location_日韩	location_日韩,日本	location_日韩,韩国	location_欧美	location_港台
# 0	0	0	1	0	0	1	0	0	1	0	...	1	0	0	0	0	0	0	0	0	1
# 1	0	0	0	1	0	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	1
# 2	0	0	0	1	0	0	1	0	0	1	...	0	0	0	0	0	0	0	0	0	1
# 3	0	1	0	0	0	1	0	0	0	1	...	0	0	0	0	0	0	0	0	0	1
# 4	0	0	0	1	0	0	1	0	1	0	...	0	0	0	0	0	0	0	0	0	1
# get_dummies one-hot处理: 将每列展开成多列,有值的为1，否则为0(根据每列的所有取值,如用户性别男 gender取值有男女 则展开为gender_女 0	gender_男 1)
df=pd.get_dummies(data[dispersed_feat])
# 离散特征数组
# one-hot数据结构
# Index(['age_0-18', 'age_19-25', 'age_26-35', 'age_36-45', 'age_46-100',
#        'gender_女', 'gender_男', 'salary_0-2000', 'salary_10000-20000',
#        'salary_2000-5000', 'salary_20000-100000', 'salary_5000-10000',
#        'province_上海', 'province_云南', 'province_内蒙古', 'province_北京',
#        'province_台湾', 'province_吉林', 'province_四川', 'province_天津',
#        'province_宁夏', 'province_安徽', 'province_山东', 'province_山西',
#        'province_广东', 'province_广西', 'province_新疆', 'province_江苏',
#        'province_江西', 'province_河北', 'province_河南', 'province_浙江',
#        'province_海南', 'province_湖北', 'province_湖南', 'province_澳门',
#        'province_甘肃', 'province_福建', 'province_西藏', 'province_贵州',
#        'province_辽宁', 'province_重庆', 'province_陕西', 'province_青海',
#        'province_香港', 'province_黑龙江', 'location_-', 'location_亚洲',
#        'location_国内', 'location_日韩', 'location_日韩,日本', 'location_日韩,韩国',
#        'location_欧美', 'location_港台'],
#       dtype='object')
one_hot_cols=df.columns
# 连续特征不做one-hot 直接存储
df[continue_feat]=data[continue_feat]

#存储交叉特征 需将user_id item_id合并
cross_feat_map=dict()
data['ui-key']=data['user_id']+'_'+data['item_id']

for _,row in data[['ui-key','score']].iterrows():
    cross_feat_map[row['ui-key']]=row['score']

cross_file=conf.cross_file
util.mkdirs(cross_file)

with open(cross_file,mode='w',encoding='utf-8') as xf:
    xf.write(str(cross_feat_map))


# 划分训练集和测试集
X_train,X_test,Y_train,Y_test=train_test_split(df,labels,test_size=0.3,random_state=2019)
# LR拟合
lr=LogisticRegression(penalty='l2',dual=False,tol=1e-4,C=1.0,fit_intercept=True,intercept_scaling=1,class_weight=None,
                      random_state=None,solver='liblinear',max_iter=100,multi_class='ovr',verbose=1,warm_start=False,n_jobs=-1)
model=lr.fit(X_train,Y_train)

W=lr.coef_
b=lr.intercept_

# array([[ 6.45567626e-04,  5.63589068e-02,  2.54601681e-02,
#          1.23244417e-01, -3.77645727e-02,  3.71198781e-02,
#          1.30824608e-01, -1.10959485e-02, -1.58755601e-02,
#         -2.16218451e-02,  1.77620051e-02,  1.98775835e-01,
#          2.17927537e-01,  1.34144375e-01, -6.26947578e-02,
#         -2.32538575e-01,  1.67873865e-01,  4.78471024e-01,
#         -3.24529453e-02,  1.47831257e-01, -1.87801815e-02,
#          2.41110429e-01,  1.55207094e-02, -1.77898265e-02,
#         -1.75692738e-01,  3.06210205e-01,  4.09225566e-01,
#          7.73440955e-02,  5.10587488e-01, -4.28190101e-02,
#         -6.19502511e-01, -9.43642212e-02, -4.54477725e-01,
#         -1.79558697e-01,  2.69611209e-01, -1.25412521e-02,
#          8.37552940e-02,  1.16868692e-01, -3.96214826e-03,
#         -2.16999925e-02,  6.17807098e-02, -1.08599257e-01,
#         -2.93965539e-01, -1.42365787e-01, -3.92993781e-01,
#         -1.63519023e-01,  9.13749407e-02,  9.25273128e-01,
#          1.90172283e-01, -1.86456911e-01,  5.42357230e-01,
#         -5.88282830e-01, -5.23372526e-01, -2.83120829e-01]])
print(W)
# array([0.16794449])
print(b)

score=lr.score(X_test,Y_test)
print(score)

#存储特征map
feat_map=dict()
for i in range(len(one_hot_cols)):
    key=one_hot_cols[i]
    feat_map[key]=i

user_feat_map_file=conf.user_feat_map_file
util.mkdirs(user_feat_map_file)
with open(user_feat_map_file,mode='w',encoding='utf-8') as onef :
    onef.write(str(feat_map))

# LR模型存储
model_dict={'W':W.tolist()[0],'b':b.tolist()[0]}

model_file=conf.model_file
util.mkdirs(model_file)
with open(model_file,mode='w',encoding='utf-8') as mf:
    mf.write(str(model_dict))
