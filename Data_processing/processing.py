import os
import pandas as pd
from mxnet import gluon
import numpy as np
from d2l import mxnet as d2l

d2l.DATA_HUB['ml-100k'] = (
    'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items

if __name__ == '__main__':
    data, num_users, num_items = read_data_ml100k()
    print(data[:1])
    print(f'shape of the data : {data.shape}, type : {type(data)}')
    print(num_users)
    print(num_items)
    print(data[-100:])
    rating_matrix = np.zeros((num_items, num_users)).T
    keys = [key for key in data]
    users = data[keys[0]].to_numpy()
    items = data[keys[1]].to_numpy()
    ratings = data[keys[2]].to_numpy()
    rating_matrix[users, items] = ratings
    # Split the data_set
