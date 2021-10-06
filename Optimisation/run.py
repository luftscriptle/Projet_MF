import numpy as np
from Optimisation import gradient_descent
from Data_processing import split_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    alpha = 0.1  # proportion of data set assigned to validation
    rating_train, user_test, index_train, movie_train, movie_test = split_data(alpha)
    n_user, n_movie = rating_train.shape
    k = 15
    init_i = np.random.random((n_user, k))
    init_u = np.random.random((n_movie, k))
    train_u, train_i, loss_vec = gradient_descent(rating_train, init_i, init_u, 0.01, eta=0.0001, n_iter=100)
    print(loss_vec)
    fig, ax = plt.subplots(1, 2)
    ax[0].semilogy(loss_vec.T[0], label='Total loss')
    ax[0].semilogy(loss_vec.T[1], label='Data loss')
    ax[1].semilogy(loss_vec.T[2],label='reg loss n°1')
    ax[1].semilogy(loss_vec.T[3], label='reg loss n°2')
    [i.legend() for i in ax]
    plt.show()

    plt.show()
