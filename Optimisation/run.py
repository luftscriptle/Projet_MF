import numpy as np
from Optimisation import gradient_descent
from Data_processing import split_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    alpha = 0.1  # proportion of data set assigned to validation
    rating_train, user_test, index_train, movie_train, movie_test = split_data(alpha)
    n_user, n_movie = rating_train.shape
    k = 10
    init_i = np.random.random((n_user, k))
    init_u = np.random.random((n_movie, k))
    train_u, train_i, loss_vec = gradient_descent(rating_train, init_i, init_u, 1, 1, 0.1)
    plt.plot(loss_vec[0])
    plt.show()
