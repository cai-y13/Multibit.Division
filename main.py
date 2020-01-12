import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from tools.quantize import quantize_model
from util.memory import ModelMemory
from tqdm import tqdm
import logging
from config import Configuration
from util.gradient import generate_grad, generate_sfc

model = models.resnet18(pretrained=True).cuda()
cfg = Configuration()

#valdir = '/home/cai-y13/imagenet/val/'
normalize = transforms.Normalize(mean=cfg.VAL_DATASET.MEAN,
                                     std=cfg.VAL_DATASET.STD)

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(cfg.VAL_DATASET.DIR, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=10, shuffle=False,
        num_workers=4, pin_memory=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0, momentum=0, weight_decay=0)

def gen_significance(model, data_loader=None, criterion=None, hierarchy=[0.0, 1.0], judge='magnitude'):
    if data_loader and criterion:
        generate_grad(model, data_loader, criterion)
    significance = generate_sfc(model, hierarchy, judge)
    return significance

def test(model, data_loader):
    correct_1 = 0
    correct_5 = 0
    total = 0 
    model.eval()
    with torch.no_grad():
        for input, target in tqdm(data_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)

            _, predict = output.topk(5, 1, True, True)
            predict = predict.t()
            correct = predict.eq(target.view(1, -1).expand_as(predict))
            correct_1 += correct[:1].view(-1).float().sum(0, keepdim=True)
            correct_5 += correct[:5].view(-1).float().sum(0, keepdim=True)
            total += target.size(0)

    print('Top1: %.3f%%' % (100. * float(correct_1) / float(total)))
    print('Top5: %.3f%%' % (100. * float(correct_5) / float(total)))

if __name__ == '__main__':
    model, fix_info = quantize_model(model)
    #significance = dict()
    #for _, (name, param) in enumerate(model.named_parameters()):
    #    significance[name] = torch.zeros(param.data.shape)
    #optimizer.zero_grad()
    significance = gen_significance(model, data_loader=val_loader, criterion=criterion, \
            hierarchy=cfg.SPLIT.HIERARCHY, judge=cfg.SPLIT.JUDGE)
    #significance = gen_significance(model, hierarchy = cfg.SPLIT.HIERARCHY, judge='random')
    model_memory = ModelMemory(model, fix_info, cfg, significance)
    model_memory.save_weights()
    model_memory.load_weights()
    
    test(model_memory.model, val_loader)
    #test(model, val_loader)
