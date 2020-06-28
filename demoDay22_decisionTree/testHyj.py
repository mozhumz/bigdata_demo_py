import pandas as pd

df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                              'Parrot', 'Parrot'],
                   'Max Speed': [380., 370., 24., 26.],
                   'Col-c':['c1','c2','c3','c4']
                   })

# print(df)

df_g=df.groupby(['Animal']).agg(list).rename(columns={'Max Speed':'Max Speed-arr','Col-c':'Col-c-arr'})
print(df_g)