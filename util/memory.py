import torch
import torch.nn as nn
import random
import math
import numpy as np
import time


class ModelMemory(object):
    def __init__(self, model, fix_info, cfg, significance):
        """
        Initialize the ModelMemory class
            - model: the deployed model that will be stored. PyTorch model file.
            - fix_info: the fixed information of the weights/bias. Format: a dictionary {'name': [bitwidth, fix_loc]}
            - splitbit (list): the split strategy for storing the weights.
            - err (dict): the error rate of the devices under different bits
            - err_inject (bool): if injecting the error
        """
        self.model = model
        self.fix_info = fix_info
        self.modelbin = dict()
        self.err_rate = cfg.ERR.RATE
        self.splitbit = cfg.SPLIT.BIT
        self.err_inject = cfg.ERR.INJECT
        self.significance = significance
    
    def save_weights(self):
        for _, (name, param) in enumerate(self.model.named_parameters()):
            if name in self.fix_info:
                start = time.time()
                bitwidth, fix_loc = self.fix_info[name]
                parambin = self.transform_param_bin(param, bitwidth, fix_loc)
                parambinsplit = self.split_param(bitwidth, parambin, self.significance[name])
                self.modelbin[name] = parambinsplit

    def load_weights(self):
        for _, (name, param) in enumerate(self.model.named_parameters()):
            if name in self.fix_info:
                start = time.time()
                bitwidth, fix_loc = self.fix_info[name]
                parambinsplit = self.modelbin[name]
                parambin = self.merge_param(parambinsplit)
                paramdec = self.transform_param_dec(parambin, fix_loc)
                paramdec = torch.from_numpy(paramdec).float()
                if param.device.type == 'cuda':
                    paramdec = paramdec.cuda()
                param.data.copy_(paramdec) 


    def transform_param_bin(self, param, bitwidth, fix_loc):
        param_np = param.data.cpu().numpy()
        shape = param_np.shape
        param_np = np.reshape(param_np, [-1,])
        parambin = [self.dec2bin(d, bitwidth, fix_loc) for d in param_np]
        parambin = np.array(parambin)
        parambin = np.reshape(parambin, shape)
        return parambin


    def transform_param_dec(self, parambin, fix_loc):
        shape = parambin.shape
        parambin = np.reshape(parambin, [-1,])
        param = [self.bin2dec(d, fix_loc) for d in parambin]
        param = np.array(param)
        param = np.reshape(param, shape)
        return param

    def dec2bin(self, decvalue, bitwidth, fix_loc=0):
        step = 2 ** (-fix_loc)
        level = round(decvalue / step)
        binstr = '0' if level >= 0 else '1'
        if level < 0:
            level += (2 ** (bitwidth - 1)) 
        for bit in range(bitwidth-2, -1, -1):
            binstr += '1' if level >= 2 ** bit else '0'
            level = level % (2 ** bit)
        return binstr

    def bin2dec(self, binvalue, fix_loc = 0):
        step = 2 ** (-fix_loc)
        bitwidth = len(binvalue)
        decvalue = 0
        for bit in range(bitwidth-2, -1, -1):
            decvalue += int(binvalue[bitwidth - 1 - bit]) * (2 ** bit)
        decvalue += int(binvalue[0]) * (-2 ** (bitwidth - 1))
        decvalue *= step
        return decvalue

    def split_param(self, bitwidth, parambin, significance):
        shape = list(parambin.shape)
        shape.append(bitwidth)
        parambin = np.reshape(parambin, [-1,])
        significance = np.reshape(significance, [-1,])
        splitparam = [self.split_bin(parambin[d], self.splitbit[int(significance[d])]) \
                for d in range(parambin.shape[0])]
        splitparam = np.array(splitparam)
        splitparam = np.reshape(splitparam, shape)
        return splitparam


    def split_bin(self, parambin, splitbit):
        splitbin = list()
        if len(parambin) != sum(splitbit):
            raise RuntimeError('The bitwidth should be aligned')
        curbit = 0
        for i in range(len(splitbit)):
            if self.err_inject:
                subbin = self.error_inject(parambin[curbit:curbit+splitbit[i]])
            else:
                subbin = parambin[curbit:curbit+splitbit[i]]
            splitbin.append(subbin)
            curbit += splitbit[i]
        if len(splitbit) < len(parambin):
            for i in range(len(parambin) - len(splitbit)):
                splitbin.append('')
        return splitbin


    def error_inject(self, binvalue):
        bitwidth = len(binvalue)
        err_rate = self.err_rate[bitwidth-1]
        #level = self.bin2dec(binvalue)
        level = 0
        for bit in range(bitwidth):
            level += int(binvalue[bit]) * (2 ** (bitwidth - bit - 1))
        rn = random.random()
        if rn < err_rate / 2 and level < 2 ** bitwidth - 1:
            level += 1
        elif err_rate / 2 <= rn < err_rate and level > 0:
            level -= 1
        #err_binvalue = self.dec2bin(level, bitwidth)
        err_binvalue = ''
        for bit in range(bitwidth-1, -1, -1):
            err_binvalue += '1' if level >= (2 ** bit) else '0'
            level = level % (2 ** bit)
        return err_binvalue
            
    def merge_param(self, splitparam):
        shape = list(splitparam.shape)
        splitn = shape[-1]
        shape = shape[:-1]
        splitparam = np.reshape(splitparam, [-1, splitn])
        mergeparam = [self.merge_bin(splitbin) for splitbin in splitparam]
        mergeparam = np.array(mergeparam)
        mergeparam = np.reshape(mergeparam, shape)
        return mergeparam


    def merge_bin(self, splitbin):
        mergebin = ''
        for i in range(len(splitbin)):
            mergebin += splitbin[i]
        return mergebin

        
def test():
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 10, 3, bias=True)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(10, 20, 3, bias=True)
            self.relu2 = nn.ReLU()
            self.fc1 = nn.Linear(100, 10, bias=True)
        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.fc1(x)
    
    def quantize(model, bitwidth = 8):
        fix_info = dict()
        for _, (name, param) in enumerate(model.named_parameters()):
            max_value = torch.max(abs(param.data))
            scale = pow(2, round(math.log2(max_value))) 
            step = 1 / (2 ** (bitwidth - 1))
            fix_loc = bitwidth - 1 - round(math.log2(max_value))
            param.data.div_(scale)
            param.data.clamp_(-1, 1-step).sub_(-1)
            param.data.div_(step).round_().mul_(step).add_(-1).mul_(scale)
            fix_info[name] = (bitwidth, fix_loc)
        return fix_info

    model = TestNet()
    fix_info = quantize(model)
    model_memory = ModelMemory(model, fix_info, (1,1,1,1,1,3), err_inject=False)
    model_memory.save_weights()
    model_memory.load_weights()

if __name__ == "__main__":
    test()
