import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def gen(n_inputs, vocab_size, max_el, pat_len, seq_len):
    inp_list = np.random.randint(0, max_el-1,
                                 size=(n_inputs, seq_len + pat_len + 1))
    inp_list[:, pat_len] = vocab_size-1
    ans_list = np.zeros((n_inputs, seq_len + pat_len + 1))
    inp_view = sliding_window_view(inp_list[:, pat_len+1:],
                                   (pat_len,), axis=-1)
    ans_list[:, 2*pat_len:] = np.all(inp_view ==
                                     inp_list[:, None, :pat_len], axis=2)
    ans_list = np.cumsum(ans_list, axis=-1, dtype='int')
    return (torch.tensor(inp_list).to('cpu'), torch.tensor(ans_list).to('cpu'))
