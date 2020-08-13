# import pandas as pd
import modin.pandas as pd
import numpy as np
import time


# import ray.dataframe as pd2

def read_csv():
    global out_dir, start_ms, priors, train, orders, products
    input_dir = 'F:\\八斗学院\\视频\\14期正式课\\00-data//'
    out_dir = input_dir + 'out/decision_tree//'
    '''
Pandas on Ray
读取数据
priors表示用户的历史购买数据
order_products__train表示用户倒数第二天的购买数据
召回中命中的为1，这个用户所有的购买过的记录作为召回商品，
train的数据为最近一天的商品，也就是从这个用户之前购买过所有商品中，
最近一天购买了属于命中了，这样模型倾向于抓住最近用户的购买需求，淡化时间久远的购买兴趣
'''
    start_ms = time.time()
    print('start_time:', start_ms)
    # 直接读取会使文件中第一列数据默认为df的index
    priors = pd.read_csv(filepath_or_buffer=input_dir + 'order_products__prior.csv', dtype={
        'order_id': np.str,
        'product_id': np.str,
        'add_to_cart_order': np.int,
        'reordered': np.int
    })
    train = pd.read_csv(filepath_or_buffer=input_dir + 'order_products__train.csv',
                        dtype={
                            'order_id': np.str,
                            'product_id': np.str,
                            'add_to_cart_order': np.int,
                            'reordered': np.int
                        })
    orders = pd.read_csv(filepath_or_buffer=input_dir + 'orders.csv',
                         dtype={
                             'order_id': np.str,
                             'user_id': np.str,
                             'eval_set': 'object',
                             'order_number': np.int16,
                             'order_dow': np.int8,
                             'order_hour_of_day': np.int,
                             'days_since_prior_order': np.float32
                         })
    products = pd.read_csv(input_dir + 'products.csv', dtype={
        'product_id': np.str,
        'order_id': np.str,
        'aisle_id': np.str,
        'department_id': np.str},
                           usecols=['product_id', 'aisle_id', 'department_id'])
    print('prior {}:{}'.format(priors.shape, ','.join(priors.columns)))
    print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
    print('train {}: {}'.format(train.shape, ', '.join(train.columns)))


# read_csv()

'''
特征处理
'''


def deal_product_feat():
    global products, priors
    prod_feat_df = pd.DataFrame()
    # 产品销量
    prod_feat_df['orders'] = priors.groupby(priors.product_id).size().astype(np.int)
    # 产品再次被购买量
    prod_feat_df['reorders'] = priors.groupby('product_id')['reordered'].sum()
    # 产品再次购买比例
    prod_feat_df['reorder_rate'] = (prod_feat_df['reorders'] / prod_feat_df['orders']).astype(np.float32)
    # 合并product的特征
    products = products.join(prod_feat_df, how='inner', on='product_id')
    # 设置product_id为index列，drop表示是否删除product_id列 inplace表示是否在原数据上修改
    products.set_index('product_id', drop=False, inplace=True)
    del prod_feat_df
    # 2 历史商品数据关联订单数据
    priors = pd.merge(priors, orders, how='inner', on='order_id')


# 1 product feat
# deal_product_feat()


def deal_user_feat():
    global users
    # 3 计算用户特征
    # 用户订单特征
    usr = pd.DataFrame()
    # 每个用户平均订单时间间隔
    usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
    # 用户订单数量
    usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int)
    # 用户商品特征
    users = pd.DataFrame()
    # 用户购买商品数量
    users['total_items'] = priors.groupby('user_id').size().astype(np.int)
    # 用户购买商品去重（set）集合
    users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
    # 用户去重后的商品数量
    users['total_distinct_items'] = users['all_products'].map(len).astype(np.int)
    # users['total_distinct_items']=users['all_products'].apply(len)
    users = users.join(usr, on='user_id')
    # 用户平均一个订单的商品数量
    users['average_basket'] = (users['total_items'] / users['nb_orders']).astype(np.float)
    print('user feat', users.shape)
    # 存储用户特征
    users.to_csv(path_or_buf=out_dir + 'users.csv')


# deal_user_feat()


def deal_userXProduct_feat():
    global userXproduct, userXproduct
    '''4用户和商品的交叉特征'''
    print('compute userXproduct f - this is long...')
    # user_id+product_id的组合key
    priors['user_product'] = priors['user_id'] + '_' + priors['product_id']
    # 存储商品和用户特征
    priors.to_csv(path_or_buf=out_dir + 'priors.csv')
    # 定义字典表 key=user_product val(1,2,3):
    # 1表示用户购买的该商品数
    # 2表示最近一个订单
    # 3表示购物车位置累加
    d = dict()
    for idx, row in priors.iterrows():
        user_product = row.user_product
        if user_product not in d:
            d[user_product] = (
                1,
                (row['order_number'], row['order_id']),
                row['add_to_cart_order']
            )
        else:
            d[user_product] = (
                d[user_product][0] + 1,
                max(d[user_product][1], (row['order_number'], row['order_id'])),
                row['add_to_cart_order'] + d[user_product][2]
            )
    # 将dict转dataframe
    userXproduct = pd.DataFrame.from_dict(d, orient='index')
    del d
    # 设置列名
    userXproduct.columns = ['nb_products', 'last_order_id', 'sum_pos']
    # 列类型转换
    userXproduct.nb_products = userXproduct.nb_products.astype(np.int)
    userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int)
    userXproduct.sum_pos = userXproduct.sum_pos.astype(np.int)
    print('user X product feat', len(userXproduct))
    del priors


# deal_userXProduct_feat()


def deal_train():
    global orders_train
    # 从orders划分训练集（用户近期的购买数据）和测试集（用户最后一天的购买数据）
    orders_train = orders[orders['eval_set'] == 'train']
    # orders_test=orders[orders['eval_set']=='test']
    # train数据以(order_id,product_id)为key inplace=True在原数据上修改 drop=False不删除原列
    train.set_index(['order_id', 'product_id'], inplace=True, drop=False)


# deal_train()


def feat_deal(selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i = 0
    for row in selected_orders.itertuples():
        i += 1
        if i % 10000 == 0:
            print('dealed rows:', i)
        order_id = row.order_id
        user_id = row.user_id
        # user_id的不重复商品集合
        # user_products=users[users.user_id==user_id].all_products
        user_products = users.all_products[user_id]
        # 产品list，即order_id的候选集
        product_list += user_products
        # 每个product对应当前的order_id,即pair(product_id,order_id)
        order_list += [order_id] * len(user_products)
        # 指定label 如果用户商品在train中那么为1
        if labels_given:
            labels += [(order_id, pid) in orders_train.index for pid in user_products]

    df = pd.DataFrame({'order_id': order_list, 'product_id': product_list}, dtype=np.str)
    labels = np.array(labels, dtype=np.int)
    del order_list
    del product_list

    # 获取user相关特征
    print("user related feat")
    df['user_id'] = df['order_id'].map(orders.user_id)
    # 用户总订单数量
    df['user_total_orders']=df['user_id'].map(users.nb_orders)
    # 用户购买的总商品数
    df['user_total_items']=df['user_id'].map(users.total_items)
    # 用户购买的去重的总商品数
    df['total_distinct_items']=df['user_id'].map(users.total_distinct_items)
    df['user_average_days_between_orders']=df['user_id'].map(users.average_days_between_orders)
    df['user_average_basket']=df['user_id'].map(users.average_basket)

    # 获取订单相关特征
    print('order related feat')
    df['order_hour_of_day']=df['order_id'].map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders

    # 商品相关特征
    print('product related feat')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    # 用户和商品的
    print('user_X_product related features')
    # 组合user_id product_id
    df['z']=df.user_id+'_'+df.product_id
    # 删除user_id
    df.drop(['user_id'],inplace=True,axis=1)
    df['UP_orders'] = df.z.map(userXproduct.nb_products)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    # 共同最后一个订单
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    # 物品在该用户订单中的平均位置
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos) / df.UP_orders).astype(np.float32)
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

if __name__ == '__main__':
    read_csv()
    deal_product_feat()
    deal_user_feat()
    deal_userXProduct_feat()
    deal_train()
    df_train,labels=feat_deal(orders_train,True)
    print('Train_columns',df_train.columns)

    # 保存结果 index=False不保存index
    df_train[f_to_use].to_csv(out_dir+'train_feat.csv',index=False)
    np.save(out_dir+'label.npy',labels)
    print('ok!')


    print('ms:',time.time()-start_ms)