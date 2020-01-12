"""
Configuration file
"""

from easydict import EasyDict as edict

class Configuration():

    #Dataset
    VAL_DATASET = edict()
    VAL_DATASET.DIR = '/home/cai-y13/imagenet/val/'
    VAL_DATASET.MEAN = [0.485, 0.456, 0.406]
    VAL_DATASET.STD = [0.229, 0.224, 0.225]

    #Error model
    ERR = edict()
    ERR.INJECT = True
    ERR.RATE = [0.0, 0.01, 0.02]

    #Split strategy
    SPLIT = edict()
    #SPLIT.BIT = [[1,1,1,1,1,3], [1,1,1,2,3], [1,2,2,3]]
    SPLIT.BIT = [[2,1,2,3], [1,1,1,2,3], [1,1,1,1,1,3]]
    #SPLIT.HIERARCHY = [0.0, 0.2, 0.6, 1.0]
    SPLIT.HIERARCHY = [0.0, 0.4, 0.8, 1.0]
    SPLIT.JUDGE = 'gradient'

