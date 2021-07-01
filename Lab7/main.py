import os
import torch
from torch.utils.data import DataLoader

from datahelper import Lab7_Dataset
from model import Generator,Discriminator
from train import train
from test import test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 100
c_dim = 300
image_shape = (64,64,3)
epochs = 500
lr_g = 0.0001
lr_d = 0.0003
batch_size = 64

if __name__ == '__main__':

    # load training data
    dataset_train = Lab7_Dataset(img_path = 'dataset/task_1/images', json_path = os.path.join('dataset/task_1', 'train.json'))
    loader_train=DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=2)

    # create generate & discriminator
    generator=Generator(z_dim,c_dim).to(device)
    # generator.load_state_dict(torch.load(os.path.join('models','epoch410_score0.69.pt'))) # c_dim = 350
    generator.load_state_dict(torch.load(os.path.join('models','epoch28_score0.79.pt'))) # c_dim = 300 # n2: 
    discrimiator=Discriminator(image_shape,c_dim).to(device)
    # discrimiator.load_state_dict(torch.load(os.path.join('models','epoch40_score0.62.pt')))
    # generator.weight_init(mean=0,std=0.02)
    discrimiator.weight_init(mean=0,std=0.02)

    # train
    train(loader_train,generator,discrimiator,z_dim,epochs,lr_g, lr_d)
    # test(loader_train,generator,discrimiator,z_dim)