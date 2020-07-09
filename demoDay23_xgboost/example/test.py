import numpy as np
import pandas as pd
# random_state = np.random.RandomState(0)
# print(random_state.randn(2,3))



df = pd.DataFrame({'col1': [1, 2,3,4], 'col2': [3, 4,5,6]})
df2 = pd.DataFrame({'col3': [31, 32,33], 'col4': [43, 44,45]})
# df3=df.join(df2)
# print(df3)
dfg=pd.DataFrame()
# df.set_index('col2',drop=False,inplace=True)
dfg['col3']=df.groupby('col1').size()
# print(dfg)

t1=(1,22)
t2=(2,11)
t3=(0,33)
t4=(4,1)

t=max(t1,t2,t3,t4)
# print(t)

# print([1] * 5)
user_products=[1,'p2','p3']
train=pd.DataFrame({'col1': [1, 2,1,2], 'col2': [3, 4,5,6]})
labels2=[]
labels2 = [ (1,product) in train.index for product in user_products]
# print(labels2)

df['col3'] = df.col1.map(df2.col3)
print(df)