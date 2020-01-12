import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def generate_grad(model, data_loader, criterion):
    model.eval()
    #for _, (name, param) in enumerate(model.named_parameters()):
    #    if param.requires_grad:
    #        param.grad.data.zero_()
    print('Generating the gradients...')
    for input, target in tqdm(data_loader):
        #if model.device.type == 'cuda':
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        cost = criterion(output, target)
        cost.backward()

def generate_sfc(model, hierarchy=[0, 0.2, 0.6, 1.0], judge='gradient'):
    significance = dict()
    for _, (name, param) in enumerate(model.named_parameters()):
        if judge == 'gradient' and param.requires_grad:
            basic = param.grad.data.view(-1)
        #elif judge == 'gradient' and not param.requires_grad:
        #    print(name)
        #    significance[name] = torch.zeros(param.data.shape)
        #    continue
        elif judge == 'magnitude' or not param.requires_grad:
            basic = param.data.view(-1)
        elif judge == 'random':
            basic = torch.rand(param.data.shape).view(-1)
        else:
            raise RuntimeError('Does not support the judgement')
        _, indices = torch.sort(torch.abs(basic), descending=True)
        wsize = indices.size(0)
        significance[name] = torch.zeros(basic.shape)
        stage = len(hierarchy) - 1
        region = [round(hierarchy[i] * wsize) for i in range(len(hierarchy))]
        for i in range(stage):
            significance[name][indices[region[i]:region[i+1]]] = i
        significance[name] = significance[name].view(param.data.shape)
        
    return significance


def test():
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 10, 3, bias=True)
        
        def forward(self, x):
            x = self.conv1(x)
            return x


    model = TestNet()
    significance = generate_sfc(model, judge='magnitude')
    for _, (name, param) in enumerate(model.named_parameters()):
        print(param)
    print(significance)

if __name__ == '__main__':
    test()
