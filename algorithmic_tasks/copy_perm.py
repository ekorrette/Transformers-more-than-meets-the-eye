import torch

def gen(batch_size,vocab_size,seq_size,device="cpu"):
    start_token = 0
    ignore_token = (vocab_size-1)
    num_special_tokens = 3
    seqs = torch.randint(num_special_tokens, vocab_size-1, (batch_size, seq_size)).to(device)

    input1 = torch.cat([
        seqs,
    ],dim=1)

    input2 = seqs[:,:-1]

    output1 = torch.cat([
        torch.ones((batch_size,seq_size),device=device)*ignore_token,
    ],dim=1)
    output2 = seqs[:,1:]

    perm = torch.randperm(seq_size-1,device=device)
    input = torch.cat([input1,input2[:,perm]],dim=1)
    output = torch.cat([output1,output2[:,perm]],dim=1)
    
    return input.long(),output.long()