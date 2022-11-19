import torch


def gen(n_inputs, vocab_size, seq_len):
    x = torch.randint(1, vocab_size-1, size=(n_inputs,))
    inp_list = torch.randint(0, vocab_size-1, size=(n_inputs, seq_len+1))
    inp_list[:, 0] = 0
    ans_list = torch.cumsum(inp_list >= x[:, None], axis=1)
    inp_list[:, 0] = x
    ans_list[:, 0] = vocab_size-1
    return (inp_list, ans_list)
