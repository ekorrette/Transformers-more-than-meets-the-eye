import torch


def gen(n_inputs, vocab_size, seq_len):
    inp_list = torch.randint(0, vocab_size-1, size=(n_inputs, 2*seq_len))
    inp_list[:, seq_len:] = 0
    ans_list = torch.sort(inp_list, dim=1)[0]
    inp_list[:, seq_len] = vocab_size - 1
    inp_list[:, seq_len+1:] = ans_list[:, seq_len:-1]
    return (inp_list, ans_list)
print(gen(2, 100, 5))
