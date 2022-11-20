
import torch
import numpy as np
from easy_transformer import EasyTransformer, EasyTransformerConfig
import easy_transformer
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

import pysvelte

def display_vectors(x,d=4):
    weights = torch.sum(x**3,dim=0)
    strongest = torch.argsort(weights)[-d:].flip(dims=(0,))
    plt.imshow(x.transpose(0,1)[strongest])
    plt.show()


def show_attention_pattern_for_input(model, inputs,outputs):
    model_cache={}
    model.cache_all(model_cache) # remove_batch_dim=True
    model(inputs)
    model.reset_hooks()
    
    pysvelte.AttentionMulti(tokens=[str(x) for x in inputs[0].tolist()], attention=model_cache['blocks.0.attn.hook_attn'][0].permute(1, 2, 0)).show()
    display_vectors(model_cache["blocks.0.mlp.hook_post"][0])
    
    pysvelte.AttentionMulti(tokens=[str(x) for x in outputs[0].tolist()], attention=model_cache['blocks.1.attn.hook_attn'][0].permute(1, 2, 0)).show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_model(testdata_generator,vocab_size,name,max_epochs=10000,n_layers=2,max_sequence_length=200,lr=1e-3):
    vocab_size = 100

    tiny_cfg = EasyTransformerConfig(
        d_model=16,
        d_head=4,
        n_heads=4,
        d_mlp=16,
        n_layers=2,
        n_ctx=200,
        act_fn="solu_ln",
        d_vocab=vocab_size,
        normalization_type="LN",
        seed=0,
    )
    tiny_model = EasyTransformer(tiny_cfg).to(device)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    eval_inputs, eval_outputs = testdata_generator(100)

    def get_loss(model, inputs, outputs):
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        
        output_logits = model(inputs, return_type="logits")

        seq_len = inputs.shape[1]

        ignore_mask = (outputs[:,:]!=(vocab_size-1)).reshape((-1,))
        
        loss = cross_entropy_loss(
            output_logits[:,:].reshape((-1,vocab_size))*ignore_mask.reshape((-1,1)),
            outputs[:,:].reshape((-1,))*ignore_mask)
        
        # loss = cross_entropy_loss(
        #     output_logits[:,:].reshape((-1,vocab_size)),
        #     outputs[:,:].reshape((-1,)))
        return loss

    def evaluate_model(model, batch_size, print_output, number_to_print=0,name=None):
        vocab_size = model.cfg.d_vocab
        inputs,outputs = testdata_generator(batch_size)
        #print(inputs,outputs)
        loss = get_loss(model,inputs,outputs)

        if print_output:
            eval_loss = get_loss(model,eval_inputs, eval_outputs)
            print(eval_loss.item())
            if name is not None:
                torch.save(model,name)

        return loss

    
    loss_history = []
    print('Start training')
    #tiny_model = torch.load("binary_xor_model")
    tiny_optimizer = torch.optim.Adam(tiny_model.parameters(), lr=lr)
    for epoch in tqdm.tqdm(range(max_epochs)):
        loss = evaluate_model(tiny_model, batch_size=200, print_output=epoch % 50 == 0, number_to_print=0, name=name)
        loss.backward()
        loss_history.append(loss.item())
        tiny_optimizer.step()
        tiny_optimizer.zero_grad()

        if loss.item()<0.1:
            break
    else:
        print("WARNING: Didn't converge within max_epochs")
    
    plt.plot(loss_history)
    plt.show()


def display_model(model,gen):
    inputs,outputs = gen(3)
    output_logits = model(inputs, return_type="logits")
    predictions = output_logits.argmax(dim=2)

    print(np.array([
        inputs[0].tolist(),
        outputs[0].tolist(),
        predictions[0].tolist()
    ]))
    show_attention_pattern_for_input(model,inputs,outputs)

    import matplotlib.pyplot as plt

    embeds = model.embed.W_E.detach().cpu()
    embeds = embeds[:,torch.argsort(embeds[0])]
    plt.imshow(torch.pca_lowrank(embeds,center=True,q=6)[0],aspect=0.02)