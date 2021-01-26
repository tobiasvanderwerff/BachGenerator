import random
import time
import math

import torch
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['init_hidden_and_cell_state', 'show_plot', 'time_since', 'count_parameters', 'save_checkpoint', 'set_seed']


def init_hidden_and_cell_state(num_layers, hidden_dim, num_directions=1, batches=1):
    h_next = torch.zeros(num_layers*num_directions, batches, hidden_dim)
    c_next = torch.zeros(num_layers*num_directions, batches, hidden_dim)
    return h_next, c_next


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def show_plot(train_losses, eval_losses, train_plot_every, eval_plot_every):
    plt.figure()
    plt.plot(np.arange(len(train_losses)) * train_plot_every, train_losses, label='training')
    plt.plot(np.arange(1, len(eval_losses) + 1) * eval_plot_every, eval_losses, color='orange', label='eval')   
    plt.xlabel('Batches processed')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '{}:{}'.format(m, int(s))
    
    
def time_since(since):
    now = time.time()
    s = now - since
    return as_minutes(s)


def save_checkpoint(net, fn):
    checkpoint = {'vocab_size': net.vocab_size, 
                  'n_hidden': net.hidden_dim,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict()
                 }
    with open(fn, 'wb') as f:
        torch.save(checkpoint, f)