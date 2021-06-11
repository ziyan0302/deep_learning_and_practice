import torch
print(torch.__version__)
print(torch.cuda.is_available())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ResNet18(nn.Module):
    def __init__(self,num_class,pretrained=False):
        # """
        # Args:
        #     num_class: #target class
        #     pretrained: 
        #         True: the model will have pretrained weights, and only the last layer's 'requires_grad' is True(trainable)
        #         False: random initialize weights, and all layer's 'require_grad' is True
        # """
        super(ResNet18,self).__init__()
        self.model=models.resnet18(pretrained=pretrained)
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad=False
        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)
        
    def forward(self,X):
        out=self.model(X)
        return out
    


def evaluate(model,loader_test,device,num_class):
    # """
    # Args:
    #     model: resnet model
    #     loader_test: testing dataloader
    #     device: gpu/cpu
    #     num_class: #target class
    # Returns:
    #     confusion_matrix: (num_class,num_class) ndarray
    #     acc: accuracy rate
    # """

    with torch.set_grad_enabled(False):
        model.eval()
        correct=0
        for images,targets in loader_test:  
            images,targets=images.to(device),targets.to(device,dtype=torch.long)
            predict=model(images)
            predict_class=predict.max(dim=1)[1]
            correct+=predict_class.eq(targets).sum().item()
        acc=100.*correct/len(loader_test.dataset)
        
    # normalize confusion_matrix

    
    return acc

class RetinopathyDataSet(Dataset):
    def __init__(self, img_path, mode):
        # """
        # Args:
        #     img_path: Root path of the dataset.
        #     mode: training/testing
            
        #     self.img_names (string list): String list that store all image names.
        #     self.labels (int or float list): Numerical list that store all ground truth label values.
        # """
        self.img_path = img_path
        self.mode = mode
        
        self.img_names=np.squeeze(pd.read_csv('train_img.csv' if mode=='train' else 'test_img.csv').values)
        self.labels=np.squeeze(pd.read_csv('train_label.csv' if mode=='train' else 'test_label.csv').values)
        assert len(self.img_names)==len(self.labels),'length not the same'
        self.data_len=len(self.img_names)
        
        self.transformations=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.3749,0.2602,0.1857),(0.2526, 0.1780, 0.1291))])
        print(f'>> Found {self.data_len} images...')
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        single_img_name=os.path.join(self.img_path,self.img_names[index]+'.jpeg')
        single_img=Image.open(single_img_name)  # read an PIL image
        img=self.transformations(single_img)
        label=self.labels[index]
        
        return img, label


num_class=5
batch_size=8
lr=5e-5
epochs=20
epochs_feature_extraction=0
epochs_fine_tuning=3
momentum=0.9
weight_decay=5e-4
Loss=nn.CrossEntropyLoss()

dataset_test=RetinopathyDataSet(img_path='data',mode='test')
loader_test=DataLoader(dataset=dataset_test,batch_size=8,shuffle=False)#,num_workers=4)

model = ResNet18(num_class=num_class)
model=model.to(device)
model.load_state_dict(torch.load(os.path.join('models','resnet18_with_pretraining81.49.pt')))

acc=evaluate(model,loader_test,device,num_class=5)
print(acc)