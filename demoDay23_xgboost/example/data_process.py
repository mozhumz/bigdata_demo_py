import pandas as pd
import numpy as np

'''
Load Data
'''
IDIR = 'D://data//data//'
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
    'order_id': np.int32,
    'product_id': np.int16,
    'add_to_cart_order': np.int16,
    'reordered': np.int8})
train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
    'order_id': np.int32,
    'product_id': np.int16,
    'add_to_cart_order': np.int16,
    'reordered': np.int8})
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
    'order_id': np.int32,
    'user_id': np.int32,
    'eval_set': 'object',
    'order_number': np.int16,
    'order_dow': np.int8,
    'order_hour_of_day': np.int8,
    'days_since_prior_order': np.float32})
products = pd.read_csv(IDIR + 'products.csv', dtype={
    'product_id': np.int16,
    'order_id': np.int32,
    'aisle_id': np.int8,
    'department_id': np.int8}, usecols=['product_id', 'aisle_id', 'department_id'])

print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))


'''
Feature Process
'''
# 1. product feature
print('computing product f')
prods = pd.DataFrame()
# 产品销量 int32
prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
# 产品再次购买量 float32
prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
# 产品再次购买比例
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
products = products.join(prods, on='product_id')
products.set_index('product_id', drop=False, inplace=True)
del prods

# 2.加入orders info到训练集priors
print('add order info to priors')
# 以orders的order_id为主key
orders.set_index('order_id', inplace=True, drop=False)
# 将所有order信息关联到priors中
priors = priors.join(orders, on='order_id', rsuffix='_')
# 有重复的列的后缀，‘_’
priors.drop('order_id_', inplace=True, axis=1)

# 3. user feature
print('computing user f')
usr = pd.DataFrame()
# 每个用户平均订单间隔时间 numpy
usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
# 用户订单数量
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)

users = pd.DataFrame()
# 用户购买商品数量
users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
# 用户购买商品去重（set）集合  apply相当于map
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
# 用户去重商品数量
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

users = users.join(usr)
del usr
# 用户平均一个订单的商品数量
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
print('user f', users.shape)

# 4. userXproduct features
print('compute userXproduct f - this is long...')
# user_id+product_id的组合key
priors['user_product'] = priors.product_id + priors.user_id * 100000

d = dict()
for row in priors.itertuples():
    z = row.user_product
    #组合key不在字典d中初始化
    if z not in d:
        # 1：user_product cnt，
        # 2：获取最近一个订单（订单顺序（最后一个订单），订单id（且订单id最大）），
        # 3：购物车位置累加
        d[z] = (1,
                (row.order_number, row.order_id),
                row.add_to_cart_order)
    else:
        d[z] = (d[z][0] + 1,
                max(d[z][1], (row.order_number, row.order_id)),
                d[z][2] + row.add_to_cart_order)

print('to dataframe (less memory)')
# 将dict转dataframe
userXproduct = pd.DataFrame.from_dict(d, orient='index')
del d
userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart.astype(np.int16)
print('user X product f', len(userXproduct))
del priors

# 5. 从orders中划分训练集和测试集
print('split orders : train, test')
test_orders = orders[orders.eval_set == 'test']
train_orders = orders[orders.eval_set == 'train']
# train数据以(order_id,product_id)为key
train.set_index(['order_id', 'product_id'], inplace=True, drop=False)


def features(selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i = 0
    for row in selected_orders.itertuples():
        i += 1
        if i % 10000 == 0: print('order row', i)
        order_id = row.order_id
        user_id = row.user_id
        # user_id的不重复商品集合
        user_products = users.all_products[user_id]
        # 产品list，即order_id的候选集
        product_list += user_products
        # 每个product对应当前的order_id,即pair(product_id,order_id)
        order_list += [order_id] * len(user_products)
        # 如果给label，即为train，此时label：如果pair（order_id,product_id）在train中为1，否则为0
        # train中给定的是对应订单，购买的产品。
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]

    df = pd.DataFrame({'order_id': order_list, 'product_id': product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list

    print('user related features')
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    df['user_average_basket'] = df.user_id.map(users.average_basket)

    print('order related features')
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders

    print('product related features')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    print('user_X_product related features')  # 10/13
    df['z'] = df.user_id * 100000 + df.product_id
    df.drop(['user_id'], axis=1, inplace=True)
    # 共同订单数量
    df['UP_orders'] = df.z.map(userXproduct.nb_orders)
    # 共同订单数量在用户所有订单中的占比 p((u,prod_i)|P,u)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    # 共同最后一个订单
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    # 物品在该用户订单中的平均位置
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    # 最后一次购买这个物品在倒数第几个订单  [1,1,1,0,1,0,1,0,0]
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    # 当前订单与最后订单时间差异（hour）
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(
        lambda x: min(x, 24 - x)).astype(np.int8)

    df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    print(df.dtypes)
    print(df.memory_usage())
    return df, labels

f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
            'user_average_days_between_orders', 'user_average_basket',
            'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
            'aisle_id', 'department_id', 'product_orders', 'product_reorders',
            'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
            'UP_average_pos_in_cart', 'UP_orders_since_last',
            'UP_delta_hour_vs_last']

df_train, labels = features(train_orders, labels_given=True)
print('Train Columns:',df_train.columns)
df_train[f_to_use].to_csv(IDIR+'train_feat.csv',index=False)
np.save(IDIR+'labels.npy',labels)

