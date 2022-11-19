import numpy as np


def gen(seed, n_inputs, seq_len):
    np.random.seed(seed)

    x = np.random.randint(0, 100, size=(n_inputs,))
    inp_list = np.random.randint(0, 100, size=(n_inputs, seq_len+1))
    inp_list[:, 0] = x
    ans_list = np.cumsum(inp_list >= x[:, None], axis=1)
    return (inp_list, ans_list)
