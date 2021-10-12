import numpy as np
from Optimisation import gradient_descent
from Data_processing import split_data, evaluate
import matplotlib.pyplot as plt

if __name__ == '__main__':
    rating_matrix = np.load('C:\\Users\\louis\\Documents\\IASD\\Projet_sciences_de_donnees\\'
                            'Projet_1\\Data_processing\\data\\rating_matrix.npy')
    alpha = 0.1  # proportion of data set assigned to validation
    rating_train, user_test, index_train, movie_train, movie_test, index_test = split_data(alpha)
    n_user, n_movie = rating_train.shape
    k = 1
    init_i = np.random.random((n_user, k))
    init_u = np.random.random((n_movie, k))
    train_u, train_i, loss_vec = gradient_descent(rating_train, init_i, init_u, 0.01, eta=0.0001, n_iter=1000)
    # print(loss_vec)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].semilogy(loss_vec.T[0], label='Total loss')
    # ax[0].semilogy(loss_vec.T[1], label='Data loss')
    # ax[1].semilogy(loss_vec.T[2],label='reg loss n°1')
    # ax[1].semilogy(loss_vec.T[3], label='reg loss n°2')
    # [i.legend() for i in ax]
    # plt.show()
    rmse = evaluate(train_u, train_i, rating_matrix, user_test, movie_test, index_test)
    print(rmse)

