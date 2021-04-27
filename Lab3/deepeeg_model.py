import torch
import torch.nn as nn
from torchvision import datasets ,transforms
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as utd
from torch.utils.tensorboard import SummaryWriter
from tensorflow import summary
from torchvision.utils import make_grid
import os
import pandas as pd
print(torch.__version__)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def read_bci_data():
    S4b_train = np.load('/content/drive/MyDrive/S4b_train.npz')
    # /content/drive/MyDrive/S4b_test.npz
    X11b_train = np.load('/content/drive/MyDrive/X11b_train.npz')
    S4b_test = np.load('/content/drive/MyDrive/S4b_test.npz')
    X11b_test = np.load('/content/drive/MyDrive/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    # print(X11b_train['signal'].shape)
    # print(train_label[:100])
    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label


train_data, train_label, test_data, test_label=read_bci_data()
dataset=utd.TensorDataset(torch.from_numpy(train_data),torch.from_numpy(train_label))
loader_train=utd.DataLoader(dataset,batch_size=64,shuffle=True)#,num_workers=4)
dataset=utd.TensorDataset(torch.from_numpy(test_data),torch.from_numpy(test_label))
loader_test=utd.DataLoader(dataset,batch_size=64,shuffle=True)#,num_workers=4)
print(f'test dataset:\n{dataset[:3]}')

class DeepEEG_Model(nn.Module):
    def __init__(self, activation=nn.ELU()):
        super(DeepEEG_Model,self).__init__()
        self.FirstConv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1,5),
                stride=(1,1),
                padding=(0,25),
                bias=True,
            ),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(2,1),
                stride=(1,1),
                padding=(0,25),
                bias=True,
            ),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )

        self.SecondConv = nn.Sequential(
            nn.Conv2d(
                in_channels=25,
                out_channels=50,
                kernel_size=(1,5),
                stride=(1,1),
                bias=True,
            ),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
        )

        self.ThirdConv = nn.Sequential(
            nn.Conv2d(
                in_channels=50,
                out_channels=100,
                kernel_size=(1,5),
                stride=(1,1),
                bias=True,
            ),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )

        self.ForthConv = nn.Sequential(
            nn.Conv2d(
                in_channels=100,
                out_channels=200,
                kernel_size=(1,5),
                stride=(1,1),
                bias=True,
            ),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.classify = nn.Linear(in_features=9800, out_features=2, bias=True)
    def forward(self, x):
        x = self.FirstConv(x)
        x = self.SecondConv(x)
        x = self.ThirdConv(x)
        x = self.ForthConv(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x


def evaluate(net,loader_test,device):
    net.eval()
    correct=0
    for idx,(data,label) in enumerate(loader_test):
        inputs=data.to(device,dtype=torch.float)
        labels=label.to(device,dtype=torch.long)
        predict=net(inputs)
        correct+=predict.max(dim=1)[1].eq(labels).sum().item()
    
    correct=100.*correct/len(loader_test.dataset)
    return correct



epochs = 800
lr = 0.00005



board = pd.DataFrame()
board['epoch'] = range(1,epochs+1)
# best_model_weighting = {'ReLU':None, 'LeakyReLU':None, 'ELU':None}
# best_evaluated_accracy = {'ReLU':0, 'LeakyReLU':0, 'ELU':0}
activations={'ReLU':nn.ReLU(),'LeakyReLU':nn.LeakyReLU(),'ELU':nn.ELU()}

for name,activation in activations.items():
  net = DeepEEG_Model().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.005)

  print(name)
  train_accuracy = []
  test_accuracy = []
  for epoch in range(epochs):
    # running_loss = 0.0
    total_loss = 0
    correct = 0
    net.train()
    for idx,(data,label) in enumerate(loader_train):
      # print("working on " , idx*256)
      inputs=data.to(device,dtype=torch.float)
      labels=label.to(device,dtype=torch.long)
      predict=net(inputs)
      loss=criterion(predict,labels)
      # print("loss: ", loss)
      total_loss+=loss.item()
      # print("total_loss: ",total_loss)
      correct+=predict.max(dim=1)[1].eq(labels).sum().item()
      """
      update
      """
      optimizer.zero_grad()
      # print("backward")
      loss.backward() 
      optimizer.step()
    total_loss/=len(loader_train.dataset)
    correct=100.*correct/len(loader_train.dataset)
    # if epoch%10==0:
    #     print(f'epcoh{epoch:>3d}  loss:{total_loss:.4f}  accuracy:{correct:.1f}%')
    train_accuracy.append(correct)
    # print("train_accuracy: " , train_accuracy)
    """
    test
    """
    net.eval()
    test_result = evaluate(net,loader_test, device)
    test_accuracy.append(test_result)

    # if test_result > 

  board[name+'_train'] = train_accuracy
  board[name+'_test'] = test_accuracy

def plot(dataframe):
    fig=plt.figure(figsize=(10,6))
    for name in dataframe.columns[1:]:
        plt.plot('epoch',name,data=dataframe)
    plt.legend()
    return fig

plot(board)
figure.savefig('deepeeg_result.png')

for column in board.columns[1:]:
    print(f'{column} max acc: {board[column].max()}')