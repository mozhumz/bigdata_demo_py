import demoDay21_recsys_music.gen_cf_data as gcd
import demoDay21_recsys_music.config as conf
# python3.7已经废弃
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 特征映射表，交叉特征，模型  输出
cross_file = conf.cross_file
user_feat_map_file = conf.user_feat_map_file
model_file = conf.model_file


# label标记,打印正负样本比例
def analysis_avg_score():
    df_user_watch = conf.user_watch()
    df_music_meta = conf.music_data()
    data = df_user_watch.merge(df_music_meta, how='inner', on='item_id')
    del df_user_watch
    del df_music_meta
    data['score'] = data.apply(lambda x: float(x['stay_seconds']) / float(x['total_timelen']),
                               axis=1)
    data = data.groupby(['user_id', 'item_id'])['score'].mean().reset_index()
    data['more_than_one'] = data['score'].apply(lambda x: 1 if x > 0.9 else 0)
    ana = data.groupby('more_than_one')['score'].count()
    print(ana)


data = gcd.user_item_score(50000, tag='avg')
# 定义label 0/1规则：希望给用户推荐的音乐，是用户能完整听完的
# 具体分析可以参考analysis_avg_score这个方法
data['label'] = data['score'].apply(lambda x: 1 if x >= 0.9 else 0)

'''
user_id,item_id,label
加入用户和item信息
'''
# user信息
user_profile = conf.user_profile()
# item信息
music_meta = conf.music_data()

# 关联用户和item的信息到data中
data = data.merge(user_profile,
                  how='inner',
                  on='user_id').merge(music_meta,
                                      how='inner',
                                      on='item_id')
# 基于字段归属，特征分类
user_feat = ['gender', 'age', 'salary', 'province']
item_feat = ['total_timelen', 'location']
item_text_feat = ['item_name', 'tags']
watch_feat = ['hours', 'stay_seconds', 'score']

# 基于字段类型，特征分类
category_feat = user_feat + ['location']
continuous_feat = ['score']

labels = data['label']
del data['label']

# # 特征处理
# 1. 离散特征one-hot处理（word2vec->embedding[continuous]）
# df数据结构 ：
# age_0-18	age_19-25	age_26-35	age_36-45	age_46-100	gender_女	gender_男	salary_0-2000	salary_10000-20000	salary_2000-5000	...	province_香港	province_黑龙江	location_-	location_亚洲	location_国内	location_日韩	location_日韩,日本	location_日韩,韩国	location_欧美	location_港台
# 0	0	0	1	0	0	1	0	0	1	0	...	1	0	0	0	0	0	0	0	0	1
# 1	0	0	0	1	0	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	1
# 2	0	0	0	1	0	0	1	0	0	1	...	0	0	0	0	0	0	0	0	0	1
# 3	0	1	0	0	0	1	0	0	0	1	...	0	0	0	0	0	0	0	0	0	1
# 4	0	0	0	1	0	0	1	0	1	0	...	0	0	0	0	0	0	0	0	0	1
df = pd.get_dummies(data[category_feat])  # 特征_特征值

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
one_hot_columns = df.columns  # ['gender_男'，'gender_女'...]
# print(df.head())
# 2.连续特征不处理直接带入[一般做离散GBDT（xgboost）叶子结点做离散化编码 GBDT+LR]
df[continuous_feat] = data[continuous_feat].astype(float)  # 转换数据类型，cast( as float)

# cross feat save(交叉特征)
# 交叉特征线要对user_id和item_id 做一个组合key
data['ui-key'] = data['user_id'].astype(str) + "_" + data['item_id'].astype(str)
cross_feat_map = dict()  # 存储到线上，这样线上也能获取到对应的特征（用户的历史行为统计类型的特征，交叉，item历史特征）
for _, row in data[['ui-key', 'score']].iterrows():
    cross_feat_map[row['ui-key']] = row['score']
# 存储交叉特征 {userid_itemid：score}
with open(cross_file, 'w') as f:
    f.write(str(cross_feat_map))

# print("样本中的X，特征") # 10条数据
# print(df.values[:10])

# 随机划分训练集train test split[0.7,0.3]
X_train, X_test, y_train, y_test = train_test_split(df.values, labels, test_size=0.3, random_state=2019)
lr = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0,
                        fit_intercept=True, intercept_scaling=1, class_weight=None,
                        random_state=None, solver='liblinear', max_iter=100,
                        multi_class='ovr', verbose=1, warm_start=False, n_jobs=-1)
'''
LogisticRegression，一共有14个参数： 
逻辑回归参数详细说明

参数说明如下：

penalty：惩罚项，str类型，可选参数为l1和l2，默认为l2。用于指定惩罚项中使用的规范。newton-cg、sag和lbfgs求解算法只支持L2规范。L1G规范假设的是模型的参数满足拉普拉斯分布，L2假设的模型参数满足高斯分布，所谓的范式就是加上对参数的约束，使得模型更不会过拟合(overfit)，但是如果要说是不是加了约束就会好，这个没有人能回答，只能说，加约束的情况下，理论上应该可以获得泛化能力更强的结果。
dual：对偶或原始方法，bool类型，默认为False。对偶方法只用在求解线性多核(liblinear)的L2惩罚项上。当样本数量>样本特征的时候，dual通常设置为False。
tol：停止求解的标准，float类型，默认为1e-4。就是求解到多少的时候，停止，认为已经求出最优解。
c：正则化系数λ的倒数，float类型，默认为1.0。必须是正浮点型数。像SVM一样，越小的数值表示越强的正则化。
fit_intercept：是否存在截距或偏差，bool类型，默认为True。
intercept_scaling：仅在正则化项为”liblinear”，且fit_intercept设置为True时有用。float类型，默认为1。
class_weight：用于标示分类模型中各种类型的权重，可以是一个字典或者’balanced’字符串，默认为不输入，也就是不考虑权重，即为None。如果选择输入的话，可以选择balanced让类库自己计算类型权重，或者自己输入各个类型的权重。举个例子，比如对于0,1的二元模型，我们可以定义class_weight={0:0.9,1:0.1}，这样类型0的权重为90%，而类型1的权重为10%。如果class_weight选择balanced，那么类库会根据训练样本量来计算权重。某种类型样本量越多，则权重越低，样本量越少，则权重越高。当class_weight为balanced时，类权重计算方法如下：n_samples / (n_classes * np.bincount(y))。n_samples为样本数，n_classes为类别数量，np.bincount(y)会输出每个类的样本数，例如y=[1,0,0,1,1],则np.bincount(y)=[2,3]。 
那么class_weight有什么作用呢？ 
在分类模型中，我们经常会遇到两类问题：
第一种是误分类的代价很高。比如对合法用户和非法用户进行分类，将非法用户分类为合法用户的代价很高，我们宁愿将合法用户分类为非法用户，这时可以人工再甄别，但是却不愿将非法用户分类为合法用户。这时，我们可以适当提高非法用户的权重。
第二种是样本是高度失衡的，比如我们有合法用户和非法用户的二元样本数据10000条，里面合法用户有9995条，非法用户只有5条，如果我们不考虑权重，则我们可以将所有的测试集都预测为合法用户，这样预测准确率理论上有99.95%，但是却没有任何意义。这时，我们可以选择balanced，让类库自动提高非法用户样本的权重。提高了某种分类的权重，相比不考虑权重，会有更多的样本分类划分到高权重的类别，从而可以解决上面两类问题。
random_state：随机数种子，int类型，可选参数，默认为无，仅在正则化优化算法为sag,liblinear时有用。
solver：优化算法选择参数，只有五个可选参数，即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear。solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是： 
liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。
saga：线性收敛的随机优化算法的的变重。
总结： 
liblinear适用于小数据集，而sag和saga适用于大数据集因为速度更快。
对于多分类问题，只有newton-cg,sag,saga和lbfgs能够处理多项损失，而liblinear受限于一对剩余(OvR)。啥意思，就是用liblinear的时候，如果是多分类问题，得先把一种类别作为一个类别，剩余的所有类别作为另外一个类别。一次类推，遍历所有类别，进行分类。
newton-cg,sag和lbfgs这三种优化算法时都需要损失函数的一阶或者二阶连续导数，因此不能用于没有连续导数的L1正则化，只能用于L2正则化。而liblinear和saga通吃L1正则化和L2正则化。
同时，sag每次仅仅使用了部分样本进行梯度迭代，所以当样本量少的时候不要选择它，而如果样本量非常大，比如大于10万，sag是第一选择。但是sag不能用于L1正则化，所以当你有大量的样本，又需要L1正则化的话就要自己做取舍了。要么通过对样本采样来降低样本量，要么回到L2正则化。
从上面的描述，大家可能觉得，既然newton-cg, lbfgs和sag这么多限制，如果不是大样本，我们选择liblinear不就行了嘛！错，因为liblinear也有自己的弱点！我们知道，逻辑回归有二元逻辑回归和多元逻辑回归。对于多元逻辑回归常见的有one-vs-rest(OvR)和many-vs-many(MvM)两种。而MvM一般比OvR分类相对准确一些。郁闷的是liblinear只支持OvR，不支持MvM，这样如果我们需要相对精确的多元逻辑回归时，就不能选择liblinear了。也意味着如果我们需要相对精确的多元逻辑回归不能使用L1正则化了。
max_iter：算法收敛最大迭代次数，int类型，默认为10。仅在正则化优化算法为newton-cg, sag和lbfgs才有用，算法收敛的最大迭代次数。
multi_class：分类方式选择参数，str类型，可选参数为ovr和multinomial，默认为ovr。ovr即前面提到的one-vs-rest(OvR)，而multinomial即前面提到的many-vs-many(MvM)。如果是二元逻辑回归，ovr和multinomial并没有任何区别，区别主要在多元逻辑回归上。 
OvR和MvM有什么不同*？* 
OvR的思想很简单，无论你是多少元逻辑回归，我们都可以看做二元逻辑回归。具体做法是，对于第K类的分类决策，我们把所有第K类的样本作为正例，除了第K类样本以外的所有样本都作为负例，然后在上面做二元逻辑回归，得到第K类的分类模型。其他类的分类模型获得以此类推。
而MvM则相对复杂，这里举MvM的特例one-vs-one(OvO)作讲解。如果模型有T类，我们每次在所有的T类样本里面选择两类样本出来，不妨记为T1类和T2类，把所有的输出为T1和T2的样本放在一起，把T1作为正例，T2作为负例，进行二元逻辑回归，得到模型参数。我们一共需要T(T-1)/2次分类。
可以看出OvR相对简单，但分类效果相对略差（这里指大多数样本分布情况，某些样本分布下OvR可能更好）。而MvM分类相对精确，但是分类速度没有OvR快。如果选择了ovr，则4种损失函数的优化方法liblinear，newton-cg,lbfgs和sag都可以选择。但是如果选择了multinomial,则只能选择newton-cg, lbfgs和sag了。
verbose：日志冗长度，int类型。默认为0。就是不输出训练过程，1的时候偶尔输出结果，大于1，对于每个子模型都输出。
warm_start：热启动参数，bool类型。默认为False。如果为True，则下一次训练是以追加树的形式进行（重新使用上一次的调用作为初始化）。
n_jobs：并行数。int类型，默认为1。1的时候，用CPU的一个内核运行程序，2的时候，用CPU的2个内核运行程序。为-1的时候，用所有CPU的内核运行程序。
总结：
优点：实现简单，易于理解和实现；计算代价不高，速度很快，存储资源低。
缺点：容易欠拟合，分类精度可能不高。
其他： 
Logistic回归的目的是寻找一个非线性函数Sigmoid的最佳拟合参数，求解过程可以由最优化算法完成。
改进的一些最优化算法，比如sag。它可以在新数据到来时就完成参数更新，而不需要重新读取整个数据集来进行批量处理。
机器学习的一个重要问题就是如何处理缺失数据。这个问题没有标准答案，取决于实际应用中的需求。现有一些解决方案，每种方案都各有优缺点。
我们需要根据数据的情况，这是Sklearn的参数，以期达到更好的分类效果。
'''
model = lr.fit(X_train, y_train)
print('w:%s, b:%s' % (lr.coef_, lr.intercept_))
print('score: %.4f' % lr.score(X_test, y_test))

# 存储特征map[key(字段名+'_'+字段值)：index]
feat_map = {}
for i in range(len(one_hot_columns)):
    key = one_hot_columns[i]
    feat_map[key] = i
print(feat_map)

# 特征映射表存储
with open(user_feat_map_file, 'w', encoding='utf-8') as ohf:
    ohf.write(str(feat_map))

# model save
model_dict = {'W': lr.coef_.tolist()[0], 'b': lr.intercept_.tolist()[0]}
with open(model_file, 'w', encoding='utf-8') as mf:
    mf.write(str(model_dict))
