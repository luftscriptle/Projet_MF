import numpy as np
import tqdm

def cost(rating_mat, i_mat, u_mat, reg_i, reg_u):
    est_rating = i_mat.dot(u_mat.T)
    term_1 = np.linalg.norm(rating_mat - est_rating)**2
    term_2 = reg_i*np.linalg.norm(i_mat)**2
    term_3 = reg_u*np.linalg.norm(u_mat)**2
    loss = term_1 + term_2 + term_3
    return loss, term_1, term_2, term_3


def grad_u(rating_mat,  i_mat, u_mat, reg_i, reg_u):
    return -rating_mat.T.dot(i_mat) + u_mat.dot(i_mat.T).dot(i_mat) + reg_u*u_mat


def grad_i(rating_mat,  i_mat, u_mat, reg_i, reg_u):
    return -rating_mat.dot(u_mat) + i_mat.dot(u_mat.T).dot(u_mat) + reg_i*i_mat


def gradient_descent(rating_mat, init_i, init_u, reg_i, reg_u, eta, n_iter=1000, ratio=1):
    step_u = eta
    step_i = eta / ratio
    i_mat = init_i
    u_mat = init_u
    loss_vec = np.zeros((n_iter, 4))
    for i in tqdm.tqdm(range(n_iter)):
        grad_1, grad_2 = grad_u(rating_mat, i_mat, u_mat, reg_i, reg_u), grad_i(rating_mat, i_mat, u_mat, reg_i, reg_u)
        u_mat = u_mat - step_u*grad_1
        i_mat = i_mat - step_i*grad_2
        loss_vec[i] = cost(rating_mat, i_mat, u_mat, reg_i, reg_u)
    return u_mat, i_mat, loss_vec
