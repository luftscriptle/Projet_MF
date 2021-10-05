import numpy as np


def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items


def split_data(alpha, rating_matrix=None):
    if rating_matrix is None:
        rating_matrix = np.load('C:\\Users\\louis\\Documents\\IASD\\Projet_sciences_de_donnees\\Projet_1\\Data_processing\\data\\rating_matrix.npy')
    users = np.load('C:\\Users\\louis\\Documents\\IASD\\Projet_sciences_de_donnees\\Projet_1\\Data_processing\\data\\users.npy')
    items = np.load('C:\\Users\\louis\\Documents\\IASD\\Projet_sciences_de_donnees\\Projet_1\\Data_processing\\data\\items.npy')
    print(rating_matrix.shape)
    n_users = rating_matrix.shape[0]
    n_test = np.int32(n_users*alpha)
    index_test = np.random.choice(np.arange(0, n_users), n_test)
    index_train = [i for i in range(n_users) if i not in index_test]
    print(index_train)


if __name__ == '__main__':
    split_data(0.1)
