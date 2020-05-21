
mid_train_data_path = '../data/mid_data/u_data.train'


def read_train_data():
    with open(mid_train_data_path,'r',encoding='utf-8') as f:
        train_dict = eval(f.read())
    return train_dict