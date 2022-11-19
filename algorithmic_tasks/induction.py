import torch

def gen(batch_size,seq_size, vocab_size,device="cpu"):
    num_special_tokens = 2
    sep_token = 1
    seqs = torch.randint(num_special_tokens, vocab_size, (batch_size, seq_size)).to(device)

    input1 = torch.cat([
        torch.ones((batch_size,1),device=device)*sep_token,
        seqs,
    ],dim=1)

    input2 = torch.cat([
        torch.ones((batch_size,1),device=device)*sep_token,
        seqs,
    ],dim=1)[:,:-1]

    output1 = torch.cat([
        torch.ones((batch_size,1+seq_size),device=device)*(vocab_size-1),
    ],dim=1)
    output2 = torch.cat([
        seqs,
        torch.ones((batch_size,1),device=device)*sep_token,
    ],dim=1)[:,:-1]

    perm = torch.randperm(seq_size,device=device)
    input = torch.cat([input1,input2[:,perm]],dim=1)
    output = torch.cat([output1,output2[:,perm]],dim=1)

    return input.long(),output.long()