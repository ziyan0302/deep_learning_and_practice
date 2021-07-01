import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

def get_iCLEVR_data(root_folder,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_folder,'test.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label

class ICLEVRLoader(data.Dataset):
    def __init__(self, mode='train'):
        self.root_folder = "/home/arg/courses/machine_learning/homework/deep_learning_and_practice/Lab7/dataset/task_1"
        self.mode = mode
        self.transformation = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])#, transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.img_list, self.label_list = get_iCLEVR_data(self.root_folder,mode)
        
        print("> Found %d images..." % (len(self.label_list)))
        self.num_classes = 24
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label_list)

    def __getitem__(self, index):
        if self.mode == "train":
            img = Image.open(os.path.join(self.root_folder, 'images', self.img_list[index])).convert("RGB")
            img = self.transformation(img)
            condition = self.label_list[index]
            condition = torch.tensor(condition)
            return img, condition
        else:
            condition = self.label_list[index]
            condition = torch.tensor(condition)
            return condition

# a = ICLEVRLoader()
# print((a[0][1]).shape)
# print(type(a[0][1]))

class Lab7_Dataset(Dataset):
    def __init__(self, img_path, json_path):
        """
        img_path: file of training images
        json_path: train.json
        """

        self.img_path = img_path
        # print(self.img_path)
        self.max_objects = 0
        datasetDir_path = '/home/arg/courses/machine_learning/homework/deep_learning_and_practice/Lab7/dataset/task_1'
        with open(os.path.join(datasetDir_path,'objects.json'),'r') as file:
            self.classes = json.load(file)
        self.numclasses = len(self.classes)
        # self.img_names = []
        self.img_conditions   = []
        with open(json_path, 'r') as file:
            list = json.load(file)
            for img_condition in list:
                self.img_conditions.append([self.classes[condition] for condition in img_condition])
            # dict = json.load(file)
            # for img_name, img_condition in dict.items():
                # self.img_names.append(img_name)
                # self.max_objects = max(self.max_objects, len(img_condition))
                # self.img_conditions.append([self.classes[condition] for condition in img_condition])
        
        self.transformations = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])#,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        # img = Image.open(os.path.join(self.img_path, self.img_names[index])).convert('RGB')
        # img = self.transformations(img)
        condition = self.int2onehot(self.img_conditions[index])
        print('condition: ',type(condition))
        return condition
    
    def int2onehot(self, int_list):
        onehot = torch.zeros(self.numclasses)
        for i in int_list:
            onehot[i] = 1.
        return onehot