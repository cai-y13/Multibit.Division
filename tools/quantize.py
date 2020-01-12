import torch
import torch.nn as nn
import math

def quantize_tensor(param, bitwidth = 8):
    max_value = torch.max(abs(param.data))
    scale = pow(2, round(math.log2(max_value)))
    step = 1 / (2 ** (bitwidth - 1))
    fix_loc = bitwidth - 1 - round(math.log2(max_value))
    param.data.div_(scale)
    param.data.clamp_(-1, 1-step).sub_(-1)
    param.data.div_(step).round_().mul_(step).add_(-1).mul_(scale)
    return fix_loc

def quantize_model(model, bitwidth = 8):
    fix_info = dict()
    for _, (name, param) in enumerate(model.named_parameters()):
        fix_loc = quantize_tensor(param, bitwidth)
        #param.data.copy_(param)
        fix_info[name] = (bitwidth, fix_loc)
    return model, fix_info


