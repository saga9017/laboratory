import numpy as np


def embedding_dim():
    return 100


def nb_epoch():
    return 30


def batch_size():
    return 30


def feat_dim():
    return 512


def w():
    return 7


def num_region():  # w*w
    return 49


def k():
    return 200


def connection():
    return "imgatt"  # it, il, lt, ik, tk, lk, itl, itk, ilk, tlk,imgatt


def which_factor():
    return "k"  # i, t, l, k


def learning_rate():
    return 0.01


def last_function():
    return "softmax"  # softmax / sigmoid

def cat_num():
    return 6


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def evaluation_factor():
    return "4factors"   # 1factor, 2factors, 3factors,  4factors, 4factors_without_weights