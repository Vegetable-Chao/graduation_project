import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

def channel_shuffle(x, groups):                                          #0.25
    batchsize, num_channels, height, width = x.data.size()             ###x.data
    channels_per_group = num_channels // groups                         ###应该可以整除
    
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()                       ##为何要这样做

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, p,K):
        super(MixedOp, self).__init__()
        self.K=K
        self.m_ops = nn.ModuleList()
        self.p = p
        self.mp = nn.MaxPool2d(2, 2)  # 加上的%
        for i in range(len(switch)):
            if switch[i]:
                primitive = PRIMITIVES[i]
                op = OPS[primitive](C//self.K, stride, False)  # 改了%
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C//self.K, affine=False))
                if isinstance(op, Identity) and p > 0:
                    op = nn.Sequential(op, nn.Dropout(self.p))  # 卷积的drop怎样运行
                self.m_ops.append(op)

    def update_p(self):  # 不太懂
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p

    def forward(self, x, weights):
        dim_2 = x.shape[1]
        xtemp = x[:, :  dim_2//self.K, :, :]
        xtemp2 = x[:,  dim_2//self.K:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self.m_ops))
        # reduction cell needs pooling before concat                           ###***
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)  # ？？？
        ans = channel_shuffle(ans, self.K)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        # except channe shuffle, channel shift also works
        return ans

# class MixedOp(nn.Module):

#     def __init__(self, C, stride, switch, p):
#         super(MixedOp, self).__init__()
#         self.m_ops = nn.ModuleList()
#         self.p = p
#         for i in range(len(switch)):
#             if switch[i]:
#                 primitive = PRIMITIVES[i]
#                 op = OPS[primitive](C, stride, False)
#                 if 'pool' in primitive:
#                     op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
#                 if isinstance(op, Identity) and p > 0:
#                     op = nn.Sequential(op, nn.Dropout(self.p))
#                 self.m_ops.append(op)
                
#     def update_p(self):
#         for op in self.m_ops:
#             if isinstance(op, nn.Sequential):
#                 if isinstance(op[0], Identity):
#                     op[1].p = self.p
                    
#     def forward(self, x, weights):
#         return sum(w * op(x) for w, op in zip(weights, self.m_ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, p,K):
        super(Cell, self).__init__()
        self.K=K
        self.reduction = reduction
        self.p = p
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count], p=self.p,K=self.K)
                self.cell_ops.append(op)
                switch_count = switch_count + 1
    
    def update_p(self):
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()

    def forward(self, s0, s1, weights,weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset+j]*self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)
        
class Ops(nn.Module):
    def __init__(self, Cin, K,choice):
        super (Ops,self).__init__()
        C=Cin//K
        self.K=K
        self.number=choice
        self.op=nn.ModuleList(
                    [nn.Sequential(
                      nn.ReLU(inplace=False),
                      nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=False),
                      nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=False),
                      nn.BatchNorm2d(C, affine=True),
                      nn.ReLU(inplace=False),
                      nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
                      nn.BatchNorm2d(C, affine=True)),
                     nn.Sequential(
                      nn.MaxPool2d(3, stride=2, padding=1),
                      nn.BatchNorm2d(C, affine=True))])
        self.mp = nn.MaxPool2d(2, 2)
        
    def forward(self,x):
        if self.K == 1:
            return self.op[self.number](x)
        else:
            dim_2 = x.shape[1]
            xtemp = x[:, :  dim_2//self.K, :, :]
            xtemp2 = x[:,  dim_2//self.K:, :, :]
            temp1 = self.op[self.number](xtemp)
            # reduction cell needs pooling before concat                           ###***
            if temp1.shape[2] == x.shape[2]:
                ans = torch.cat([temp1, xtemp2], dim=1)
            else:
                ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)  # ？？？
            ans = channel_shuffle(ans, self.K)
            #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
            # except channe shuffle, channel shift also works
            return ans
        
            
        
        

class GDAS_Reduction_Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, p,K):
    #除了通道参数其他没有作用，传入是为了net部分代码方便。
    super(GDAS_Reduction_Cell, self).__init__()
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self.multiplier  = multiplier

    self.reduction = True
    
    self.ops1 = nn.ModuleList([Ops(C,K,0),Ops(C,K,0)])
    self.ops2 = nn.ModuleList([Ops(C,K,1),Ops(C,K,1)])
    

  def forward(self, s0, s1, weights,weights2):                
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)


    X0 = self.ops1[0] (s0)
    X1 = self.ops1[1] (s1)
    # if self.training and drop_prob > 0.:
    #   X0, X1 = drop_path(X0, drop_prob), drop_path(X1, drop_prob)

    #X2 = self.ops2[0] (X0+X1)
    X2 = self.ops2[0] (s0)
    X3 = self.ops2[1] (s1)
    # if self.training and drop_prob > 0.:
    #   X2, X3 = drop_path(X2, drop_prob), drop_path(X3, drop_prob)
    return torch.cat([X0, X1, X2, X3], dim=1)




class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, switches_normal=[], switches_reduce=[], p=0.0,K=4,use_baidu=True,use_EN=False):
        super(Network, self).__init__()
        self.weights_normal=0
        self.weights_reduce=0
        self.weights2_normal=0
        self.weights2_reduce=0
        self.K=K
        self.use_EN=use_EN
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.p = p
        self.switches_normal = switches_normal
        switch_ons = []
        for i in range(len(switches_normal)):                  #？？？
            ons = 0
            for j in range(len(switches_normal[i])):
                if switches_normal[i][j]:
                    ons = ons + 1
            switch_ons.append(ons)
            ons = 0
        self.switch_on = switch_ons[0]

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
                if use_baidu:
                    cell = GDAS_Reduction_Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.p,self.K)
                else:
                    cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_reduce, self.p,self.K)
            else:
                reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal, self.p,self.K)
#            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, input):
        self.weights_normal = F.sigmoid(self.alphas_normal) #注意改名了
        self.weights_reduce = F.sigmoid(self.alphas_reduce)
        if self.use_EN:
            self.weights2_normal = self.get_weights2()[0]
            self.weights2_reduce = self.get_weights2()[1]
        else:
            self.weights2_normal = torch.ones_like(self.betas_normal)
            self.weights2_reduce = torch.ones_like(self.betas_reduce)
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights=self.weights_reduce
                weights2=self.weights2_reduce
            else:
                weights=self.weights_normal
                weights2=self.weights2_normal
            s0, s1 = s1, cell(s0, s1, weights,weights2)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits
        
    def get_weights2(self):
        n = 3
        start = 2
        weights2_normal = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weights2_normal = torch.cat([weights2_normal,tw2],dim=0)
        n = 3
        start = 2
        weights2_reduce = F.softmax(self.betas_reduce[0:2], dim=-1)
        for i in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            start = end
            n += 1
            weights2_reduce = torch.cat([weights2_reduce,tw2],dim=0)
        return weights2_normal,weights2_reduce
        


    def update_p(self):
        for cell in self.cells:
            ###改动了
            if cell.reduction == False:
                cell.p = self.p
                cell.update_p()
    
    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i)) 
        num_ops = self.switch_on                                      #？？？
        self.alphas_normal = nn.Parameter(torch.zeros(k, num_ops))
        self.alphas_reduce = nn.Parameter(torch.zeros(k, num_ops))
        self.betas_normal = nn.Parameter(torch.FloatTensor(1e-3*torch.randn(k)))
        self.betas_reduce = nn.Parameter(torch.FloatTensor(1e-3*torch.randn(k)))

        
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce,
        ]
    
    
    def arch_parameters(self):
        return self._arch_parameters


