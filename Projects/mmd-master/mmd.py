import theano.tensor as T


def mmd_full(x_t, y_t, alpha=0.5):
    """ Implementation of the full kernel MMD statistic (gaussian kernel)"""
    N = x_t.shape[1]
    M = y_t.shape[1]

    term1 = T.mean(T.exp(-0.5 * (1 / alpha) * T.square(T.repeat(x_t, N) - T.tile(x_t, N))))
    term2 = T.mean(T.exp(-0.5 * (1 / alpha) * T.square(T.repeat(x_t, M) - T.tile(y_t, N))))
    term3 = T.mean(T.exp(-0.5 * (1 / alpha) * T.square(T.repeat(y_t, M) - T.tile(y_t, M))))
    return term1 - 2 * term2 + term3


def mmd_approx(x_t, y_t, alpha=0.5):
    """ Implementation of the linear time approximation to the gaussian kernel MMD statistic"""
    M = x_t.shape[1] // 2
    odd_x = x_t[:, ::2]
    even_x = x_t[:, 1::2]

    odd_y = y_t[:, ::2]
    even_y = y_t[:, 1::2]

    term1 = 2 * T.mean(T.exp(-0.5 * (1 / alpha) * T.square(odd_x - even_x)))  # k(x_{2i-1}, x_{2i})
    term2 = 2 * T.mean(T.exp(-0.5 * (1 / alpha) * T.square(odd_y - even_y)))  # k(y_{2i-1}, y_{2i})
    term3 = 2 * T.mean(T.exp(-0.5 * (1 / alpha) * T.square(odd_x - even_y)))  # k(x_{2i-1}, y_{2i})
    term4 = 2 * T.mean(T.exp(-0.5 * (1 / alpha) * T.square(even_x - odd_y)))  # k(x_{2i}, y_{2i-1})
    return term1 + term2 - term3 - term4
