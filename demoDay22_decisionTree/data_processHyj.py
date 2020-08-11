import pandas as pd
import numpy as np

input_dir = 'F:\\八斗学院\\视频\\14期正式课\\00-data//'

'''
读取数据
priors表示用户的历史购买数据
order_products__train表示用户倒数第二天的购买数据
'''
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

print('prior {}:{}'.format(priors.shape,','.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))

'''
特征处理
'''
# 1 product feat
prod_feat_df=pd.DataFrame()
# 产品销量
prod_feat_df['orders']=priors.groupby(priors.product_id).size().astype(np.int)
# 产品再次被购买量
prod_feat_df['reorders']=priors.groupby('product_id')['reordered'].sum()
# 产品再次购买比例
prod_feat_df['reorder_rate']=(prod_feat_df['reorders']/prod_feat_df['orders']).astype(np.float32)
# 合并product的特征
products=products.join(prod_feat_df,how='inner',on='product_id')
# 设置product_id为index列，drop表示是否删除product_id列 inplace表示是否在原数据上修改
products.set_index('product_id',drop=False,inplace=True)
# del prod_feat_df
priors=pd.merge(priors,orders,how='inner',on='order_id')


priors['order_id']=priors['order_id'].astype(np.int32)
orders['order_id']=orders['order_id'].astype(np.int32)