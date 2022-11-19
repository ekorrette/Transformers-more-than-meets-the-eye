import torch


def gen(n_inputs, vocab_size, seq_len):
    inp_list = 2 * torch.randint(0, 2, size=(n_inputs, seq_len))
    ans_list = torch.cumsum(inp_list-1, axis=1)
    ans_list = torch.sign(ans_list) + 1
    return (inp_list, ans_list)
