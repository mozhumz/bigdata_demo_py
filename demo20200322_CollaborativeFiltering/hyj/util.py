mid_train_data_path='../data/mid_data/u_data.train'

def get_train_data():
    data=dict()
    with open(mid_train_data_path,mode='r',encoding='utf-8') as f:
        data= eval(f.read())

    return data

