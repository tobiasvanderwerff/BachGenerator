import copy
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.util_TMP import init_hidden_and_cell_state

      
def restore_state(net, params_state, optimizer, optimizer_state, scheduler=None, scheduler_state=None):
    optimizer.load_state_dict(optimizer_state)
    net.load_state_dict(params_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)

        
class LRFinder:
    """
    Learning rate finder to be used in combination with the 1cycle policy by Leslie Smith.
    See https://sgugger.github.io/the-1cycle-policy.html for more information.
    """
    
    def __init__(self, net, optimizer, criterion, scheduler=None):
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.losses = None
        self.lrs = None
    
    def plot(self, start=5, end=-1):
        if self.lrs is None or self.losses is None:
            raise AttributeError("lrs and losses not set. Run lr_find first.")
        plt.figure()
        plt.plot(self.lrs[start:end], self.losses[start:end])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()
        
    def lr_find_lstm(self, data, bs, begin_lr=0.001, end_lr=10, beta=0.98, num_it=100, device='cuda:0'):
        """
        Run the 1cycle learning rate finder.
        
        Inputs:
        - data: input data, as dataloader or tuple (X, Y)
        - bs: batchsize 
        """
        
        # Save state of the network to restore afterwards.
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        params_state = copy.deepcopy(self.net.state_dict())
        if self.scheduler is not None:
            scheduler_state = copy.deepcopy(self.scheduler.state_dict())
        else: 
            scheduler_state = None
            
        h_next, c_next = init_hidden_and_cell_state(self.net.n_layers, self.net.hidden_dim, self.net.n_directions, bs)
        h_next, c_next = h_next.to(device), c_next.to(device)
        
        self.net.train()
        losses, lrs = [], []
        moving_average, best_loss, iter_count = 0, 0, 0
        mult = (end_lr / begin_lr) ** (1 / num_it)
        lr = begin_lr
        while iter_count < num_it:
            for dat in data:
                if iter_count >= num_it:
                    break
                
                inputs, targets = dat[0], dat[1]
                if not isinstance(data, DataLoader):
                    inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
                    inputs = inputs.unsqueeze(0)
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                self.optimizer.zero_grad()
                for i in range(len(self.optimizer.param_groups)):
                    self.optimizer.param_groups[i]['lr'] = lr  # Set new learning rate for each parameter group

                logits, (h_next, c_next) = self.net(inputs, (h_next, c_next))
                loss = self.criterion(logits, targets)

                moving_average = beta * moving_average + (1-beta) * loss.item() 
                smoothed_average = moving_average / (1 - beta**(iter_count+1))  # bias-corrected version of the average

                if smoothed_average < best_loss or iter_count == 0:
                    best_loss = smoothed_average
                if smoothed_average > 4 * best_loss and iter_count > 0:  # stop early if loss explodes
                    self.lrs, self.losses = lrs, losses
                    restore_state(self.net, params_state, self.optimizer, optimizer_state, self.scheduler, scheduler_state)
                    return
                
                # After forward pass: requires_grad = True for h and c. Reset this.
                (h_next, c_next) = tuple(a.data for a in (h_next, c_next))

                loss.backward()
                self.optimizer.step()

                lrs.append(lr)
                losses.append(smoothed_average)
                lr *= mult
                iter_count += 1
        self.lrs, self.losses = lrs, losses
        restore_state(self.net, params_state, self.optimizer, optimizer_state, self.scheduler, scheduler_state)