import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

# STE
class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional

""" 1. for pretrain"""
class Attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(Attention2d, self).__init__()
        #assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class DynConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=0.5, init_weight=True):
        super(DynConv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = GPM2d(in_planes, out_planes, ratio, temperature, K,
                        kernel_size=kernel_size, extract_type='avg') 
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        self.num_out = 0
        self.num_full = 0

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        attn = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)

        aggregate_weight = (self.weight.unsqueeze(0) * attn).sum(1).view(-1, self.in_planes, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))

        return output


""" 2. for dynamic convolution"""
class GPM2d(nn.Module):
    def __init__(self, in_planes, out_planes, ratios, temperature, num_pattern, kernel_size, extract_type='avg', init_weight=True):
        super(GPM2d, self).__init__()
        #assert temperature%3==1
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = num_pattern
        self.temperature = temperature

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.num_pattern = num_pattern
        self.kernel_size = kernel_size
        
        # Pattern softmax attention
        self.psa = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, num_pattern*out_planes, 1, bias=True)
            )

        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):

        psa = self.psa(x).view(x.size(0), self.num_pattern, self.out_planes, 1, 1, 1)

        return F.softmax(psa/self.temperature, 1)


class DPPConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, num_pattern=4, temperature=0.5, init_weight=True):
        super(DPPConv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.num_pattern = num_pattern
        self.attention = GPM2d(in_planes, out_planes, ratio, temperature, num_pattern,
                        kernel_size=kernel_size, extract_type='avg')
        self.weight = nn.Parameter(torch.randn(num_pattern, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)

        
        # threshold
        self.th = 0
        self.threshold = nn.Parameter(self.th * torch.ones(1, num_pattern, out_planes, 1, 1, 1)) #bs, p, o, i, k, k
        self.step = BinaryStep.apply
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_pattern, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()
            with torch.no_grad():
                self.threshold.data.fill_(0.)


        self.num_out = torch.tensor([0])
        self.num_full = torch.tensor([0])

    def _initialize_weights(self):
        for i in range(self.num_pattern):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        bs, in_planes, height, width = x.size()

        # 1. Generate Mask
        attn = self.attention(x)
        mask = self.step(F.sigmoid((self.weight.unsqueeze(0).abs() - self.threshold).mean(dim=(2,3))) - 0.5).view(1, self.num_pattern, 1, 1, self.kernel_size, self.kernel_size)

        # 2. Generate dynamic weights
        aggregate_weight = (self.weight.unsqueeze(0) * mask * attn).sum(1).view(-1, self.in_planes, self.kernel_size, self.kernel_size) 
        x = x.view(1, -1, height, width) 

        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups * bs)

        output = output.view(bs, self.out_planes, output.size(-2), output.size(-1))

        if not self.training:
            sparse_mask = (mask*attn).sum(1)
            self.num_out += sparse_mask.numel()*in_planes*output.size(-2)*output.size(-1)
            self.num_full += sparse_mask[sparse_mask>0].numel()*in_planes*output.size(-2)*output.size(-1)
            
            # attention calculation
            self.num_keep += self.kernel_size*self.kernel_size*self.in_planes*self.out_planes*self.num_pattern
            self.num_all += self.kernel_size*self.kernel_size*self.in_planes*self.out_planes*self.num_pattern

        return output

""" 3. for fine-tuning"""
class DynConv2d2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=0.5, init_weight=True):
        super(DynConv2d2, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = GPM2d(in_planes, out_planes, ratio, temperature, K,
                        kernel_size=kernel_size, extract_type='avg') 
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        self.th = 0
        self.threshold = nn.Parameter(self.th * torch.ones(1, 4, out_planes, 1, 1, 1)) #bs, p, o, i, k, k
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        self.num_out = 0
        self.num_full = 0

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        attn = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)

        aggregate_weight = (self.weight.unsqueeze(0) * attn).sum(1).view(-1, self.in_planes, self.kernel_size, self.kernel_size)

        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))

        return output
