import torch
import torch.nn as nn
import torchvision.models as models

'''===============================================================
1. Title:     
DLP spring 2020 Lab6 classifier
2. Purpose:
For computing the classification accruacy.
3. Details:
The model is based on ResNet18 with only changing the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled 
64x64 images from 32x32 images.
It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.
4. How to use
You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]
==============================================================='''

class EvaluationModel():
    def __init__(self):
        checkpoint = torch.load('classifier_weight.pth')
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, 24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()
        self.resnet18.eval()
        self.classnum = 24
    
    def compute_accuracy(self, out, onehot_labels):
        batch_size = out.size(0)
        accuracy = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total +=k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    accuracy +=1
        return accuracy/total

    def eval(self, images, labels):
        with torch.no_grad():
            out = self.resnet18(images) #images size:(B,3,64,64)
            accuracy = self.compute_accuracy(out.cpu(),labels.cpu())
            return accuracy

