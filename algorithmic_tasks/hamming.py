import torch

def num_to_bin(num,binsize):
    num = num.clone()
    out = torch.zeros((len(num),binsize)).long()
    for i in range(binsize):
        out[:,i] = num&1
        num >>= 1
    return out

def gen(n_inputs, vocab_size, seq_len=16,code_len=4):
    bits = torch.randint(0, 2, size=(n_inputs,seq_len))
    indx = (torch.arange(0, seq_len).reshape((1,-1))*bits).long()

    res = torch.zeros((n_inputs,)).long()
    for i in range(seq_len):
        res = torch.bitwise_xor(res,indx[:,i])
    
    
    b = torch.randint(0, 1<<seq_len, size=(n_inputs,))
    seq = torch.cat([
        bits,
        num_to_bin(res,code_len),
    ],dim=1)
    
    inputs = seq[:,:-1].clone()
    outputs = seq[:,1:].clone()
    outputs[:,:seq_len-1] = (vocab_size-1)

    return (inputs.long(), outputs.long())
