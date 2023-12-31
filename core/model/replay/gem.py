# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import quadprog

#from .common import MLP, ResNet18

# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    '''
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    '''
    #offset1 = task * nc_per_task
    offset1 = 0
    offset2 = (task + 1) * nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            #print("ok")
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    
    #memories_np = memories.t().double().numpy()
    #gradient_np = gradient.contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class GEM(nn.Module):
    '''
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
    '''
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__()
        #nl, nh = kwargs.n_layers, kwargs.n_hiddens
        #self.margin = kwargs["memory_strength"]#?
        
        #self.net = ResNet18(n_outputs)
        #add defination in 
        self.net = backbone
        #classifier
        self.classifier = nn.Linear(feat_dim, num_class)

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = num_class
        n_tasks = kwargs["n_task"]

        #self.opt = optim.SGD(self.parameters(), args.lr)

        #memory_size
        self.n_memories = (int)(kwargs["n_memories"]/n_tasks)
        #self.gpu = args.cuda
        self.device = kwargs['device']
        #n_inputs = 32#input_size
        d=3
        h=32
        w=32

        # allocate episodic memory
        #self.memory_data = torch.FloatTensor(
           # n_tasks, self.n_memories, n_inputs)

        #n_task:100
        #self.n_memories :500(buffer)
        #
        self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, d,h,w)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        '''
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()
        '''
        self.memory_data.to(self.device)
        self.memory_labs.to(self.device)
        
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        '''
        if args.cuda:
            self.grads = self.grads.cuda()
        '''
        self.grads = self.grads.to(self.device)
        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

        self.nc_per_task = 20
        self.t = 0
        '''
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
        '''

    def forward(self, x, t):
        #print(x.device)
        x = x.to(self.device)
        output = self.classifier(self.net(x)['features'])
        '''
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        '''
        #offset1 = int(t * self.nc_per_task)
        offset1 = 0
        offset2 = int((t + 1) * self.nc_per_task)
        if offset1 > 0:
            #output[:, :offset1].data.fill_(-10e10)
            output[:, :offset1].fill_(-10e10)
        if offset2 < self.n_outputs:
            #output[:, offset2:self.n_outputs].data.fill_(-10e10)
            output[:, offset2:self.n_outputs].fill_(-10e10)
        #print(output.size())
        return output

    #def observe(self, x, t, y):
    def observe(self, data):
        '''
        修改observe的传入参数为batch(data),并表示原先的参数 x,y,t(task)
        '''
        # get data and labels
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        # update memory
        if self.t != self.old_task:
            self.observed_tasks.append(self.t)
            self.old_task = self.t

        # Update ring buffer storing examples from current task
        #bsz = y.data.size(0)
        bsz = y.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        '''
        self.memory_data[self.t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        '''
        #print(self.memory_data[self.t, self.mem_cnt: endcnt].size())
        #print(x[: effbsz].size())
        self.memory_data[self.t, self.mem_cnt: endcnt].copy_(
            x[: effbsz])
        if bsz == 1:
            self.memory_labs[self.t, self.mem_cnt] = y.data[0]
        else:
            '''
            self.memory_labs[self.t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
            '''
            self.memory_labs[self.t, self.mem_cnt: endcnt].copy_(
                y[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        
        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                #offset1, offset2 = compute_offsets(past_task, self.nc_per_task)
                offset1, offset2 = compute_offsets(self.t, self.nc_per_task)
                
                
                '''
                print(self.forward(
                        self.memory_data[past_task],
                        past_task)[:, offset1: offset2].device)
                
                print((self.memory_labs[past_task] - offset1).device)
                '''
                '''
                print(self.forward(
                        self.memory_data[past_task],
                        past_task)[:, offset1: offset2])
                print(self.memory_labs[past_task] - offset1)
                
                
                _, predicted = torch.max(self.forward(
                        self.memory_data[past_task],
                        past_task)[:, offset1: offset2], 1)  
                
                
                print(predicted)
                print(self.memory_labs[past_task] - offset1)
                '''
                #print(self.memory_data[past_task])
                #print(self.memory_labs[past_task] - offset1)
                '''
                ptloss = self.ce(
                    self.forward(
                        self.memory_data[past_task],
                        past_task)[:, offset1: offset2],
                    (self.memory_labs[past_task] - offset1).to(self.device))
                '''
                '''
                ptloss = self.ce(
                    self.forward(
                        self.memory_data[past_task],
                        self.t)[:, offset1: offset2],
                    (self.memory_labs[past_task] - offset1).to(self.device))
                '''
                ptloss = self.ce(
                    self.forward(
                        self.memory_data[past_task],
                        self.t),
                    (self.memory_labs[past_task] - offset1).to(self.device))
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)
        

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(self.t, self.nc_per_task)
        #output = self.forward(x, self.t)[:, offset1: offset2]
        output = self.forward(x, self.t)
        _, predicted = torch.max(output, 1)  
        #print(predicted)
        #print(y)
        correct = (predicted == (y - offset1)).sum().item()  
        total = y.size(0) 
        acc = correct / total 

        loss = self.ce(output, y - offset1)
        loss.backward()

        # check if gradient violates constraints
        # 不满足约束会更新梯度
        #to-doS
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, self.t)
            #indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
            #    else torch.LongTensor(self.observed_tasks[:-1])
            indx = torch.LongTensor(self.observed_tasks[:-1])
            indx = indx.to(self.device)
            self.grads = self.grads.to(self.device)
            result = self.grads.index_select(1, indx)
            dotp = torch.mm((self.grads[:, self.t].unsqueeze(0)).to(self.device),
                            result)
            dotp.to(self.device)
            if (dotp < 0).sum() != 0:
                #project2cone2(self.grads[:, self.t].unsqueeze(1),
                              #self.grads.index_select(1, indx), self.margin)
                project2cone2(self.grads[:, self.t].unsqueeze(1),
                              self.grads.index_select(1, indx))
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, self.t],
                               self.grad_dims)
        #self.opt.step()

        return output, acc, loss
    
    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        offset1, offset2 = compute_offsets(self.t, self.nc_per_task)
        #output = self.forward(x, self.t)[:, offset1: offset2]
        output = self.forward(x, self.t)
        _, predicted = torch.max(output, 1)  
        correct = (predicted == (y - offset1)).sum().item()  
        total = y.size(0) 
        acc = correct / total 
        return predicted, acc
    

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        self.t += 1
        self.mem_cnt = 0
        #print(self.t)
