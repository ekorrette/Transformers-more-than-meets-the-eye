import torch

def num_to_bin(num,binsize):
    num = num.clone()
    out = torch.zeros((len(num),binsize)).long()
    for i in range(binsize):
        out[:,i] = num&1
        num >>= 1
    return out

def gen(n_inputs, vocab_size, seq_len):
    a = torch.randint(0, 1<<seq_len, size=(n_inputs,))
    b = torch.randint(0, 1<<seq_len, size=(n_inputs,))
    seq = torch.cat([
        torch.ones((n_inputs,1))*2,
        num_to_bin(a,seq_len),
        torch.ones((n_inputs,1))*3,
        num_to_bin(b,seq_len),
        torch.ones((n_inputs,1))*4,
        num_to_bin(a,seq_len)^num_to_bin(b,seq_len),
    ],dim=1)
    
    inputs = seq[:,:-1].clone()
    outputs = seq[:,1:].clone()
    outputs[:,:-seq_len] = (vocab_size-1)

    return (inputs.long(), outputs.long())
