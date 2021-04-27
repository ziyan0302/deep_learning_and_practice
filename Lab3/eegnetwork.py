import torch
import torch.nn as nn
from torchvision import datasets ,transforms
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow
import torch.optim as optimwqa
from torch.autograd import Variable
import torch.utils.data as utd
from torch.utils.tensorboard import SummaryWriter
from tensorflow import summary
from torchvision.utils import make_grid
import os
import pandas as pd
import timeit
print(torch.__version__)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class EEG_Model(nn.Module):
    def __init__(self, activation=nn.ELU()):
        super(EEG_Model,self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1,51),
                stride=(1,1),
                padding=(0,25),
                bias=False,
            ),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2,1),
                groups=16,
                stride=(1,1),
                bias=False
            ),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.25)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1,15),
                stride=(1,1),
                padding=(0,7),
                bias=False,
            ),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=0.25),
        )

        self.classify = nn.Linear(736, 2)#, bias=True)
    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x

def read_bci_data():
    S4b_train = np.load('/content/drive/MyDrive/S4b_train.npz')
    X11b_train = np.load('/content/drive/MyDrive/X11b_train.npz')
    S4b_test = np.load('/content/drive/MyDrive/S4b_test.npz')
    X11b_test = np.load('/content/drive/MyDrive/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

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



# train_data, train_label, test_data, test_label=read_bci_data()
train_data, train_label, test_data, test_label =read_bci_data()
dataset=utd.TensorDataset(torch.from_numpy(train_data),torch.from_numpy(train_label))
loader_train=utd.DataLoader(dataset,batch_size=256,shuffle=True)#,num_workers=4)
dataset=utd.TensorDataset(torch.from_numpy(test_data),torch.from_numpy(test_label))
loader_test=utd.DataLoader(dataset,batch_size=256,shuffle=True)#,num_workers=4)
print(f'test dataset:\n{dataset[:3]}')



epochs = 10
lr = 0.0004


start_time = timeit.default_timer()

board = pd.DataFrame()
board['epoch'] = range(1,epochs+1)
criterion = nn.CrossEntropyLoss()

# best_model_weighting = {'ReLU':None, 'LeakyReLU':None, 'ELU':None}
# best_evaluated_accracy = {'ReLU':0, 'LeakyReLU':0, 'ELU':0}
# activations={'ReLU':nn.ReLU()}
activations={'ReLU':nn.ReLU(),'LeakyReLU':nn.LeakyReLU(),'ELU':nn.ELU()}

for name,activation in activations.items():
  net = EEG_Model(activation)
  net.to(device)
  
  optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)
  print(name)
  train_accuracy = []
  test_accuracy = []
  for epoch in range(epochs):
    # running_loss = 0.0
    total_loss = 0
    correct = 0
    test_correct=0
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
      # print('train: ',correct)

      """
      update
      """
      optimizer.zero_grad()
      # print("backward")
      loss.backward() 
      optimizer.step()
    total_loss/=len(loader_train.dataset)
    correct=100.*correct/len(loader_train.dataset)
    
    
    train_accuracy.append(correct)
    # print("train_accuracy: " , train_accuracy)
    """
    test
    """
    net.eval()
    # test_result = evaluate(net,loader_test, device)
    
    for idx,(data_t,label_t) in enumerate(loader_test):
        inputs_t=data_t.to(device,dtype=torch.float)
        labels_t=label_t.to(device,dtype=torch.long)
        predict_t=net(inputs_t)
        test_correct+=predict_t.max(dim=1)[1].eq(labels_t).sum().item()
        # print('test: ',test_correct)
    
    test_correct=100.*test_correct/len(loader_test.dataset)
    test_accuracy.append(test_correct)
    # if epoch%10==0:
        # print(f'epcoh{epoch:>3d}  loss:{total_loss:.4f}  train_accuracy:{correct:.1f}% test_accuracy:{test_result:1f}%')



  board[name+'_train'] = train_accuracy
  board[name+'_test'] = test_accuracy

end_time = timeit.default_timer()
print("Run time %.2f s" % (end_time-start_time))


def plot(dataframe):
    fig=plt.figure(figsize=(10,6))
    for name in dataframe.columns[1:]:
        plt.plot('epoch',name,data=dataframe)
    plt.legend()
    return fig


plot(board)
figure.savefig('eeg_result.png')


for column in board.columns[1:]:
    print(f'{column} max acc: {board[column].max()}')