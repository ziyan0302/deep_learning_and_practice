import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset import ICLEVRLoader, Lab7_Dataset


datasetDir_path = '/home/arg/courses/machine_learning/homework/deep_learning_and_practice/Lab7/dataset/task_1'
datasetImgDir_path = '/home/arg/courses/machine_learning/homework/deep_learning_and_practice/Lab7/dataset/task_1/images'
testset = Lab7_Dataset(img_path = datasetImgDir_path, json_path = os.path.join(datasetDir_path,'test.json'))
print(testset)
print('ok')