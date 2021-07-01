import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_shape, c_dim): #c_dim is not necessary?
        super(Discriminator, self).__init__()
        self.height, self.width, self.channel = img_shape
        self.conditionExpand = nn.Sequential(
            nn.Linear(24, self.height*self.width),
            nn.LeakyReLU()
        )
        kernel_size = (4, 4)
        channels = [4, 64, 128, 256, 512]
        for i in range(1, len(channels)):
            setattr(
                self,'conv'+str(i),nn.Sequential(
                    nn.Conv2d(channels[i-1], channels[i], kernel_size, stride = (2,2), padding = (1,1)),
                    nn.BatchNorm2d(channels[i]),
                    nn.LeakyReLU()
                )
            )
        self.conv5 = nn.Conv2d(512, 1, kernel_size, stride = (1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, c):
        """
        X: (batch_size, 3, 64, 64) tensor
        c: (batch_size, 24) tensor
        return: (batch_size) tensor
        """
        c = self.conditionExpand(c).view(-1,1,self.height,self.width) # (N,64,64)
        out = torch.cat((X,c),dim = 1) #become(N,4,64,64)
        out = self.conv1(out) #(N,64,32,32)
        out = self.conv2(out) #(N,128,16,16)
        out = self.conv3(out) #(N,256,8,8)
        out = self.conv4(out) #(N,512,4,4)
        out = self.conv5(out) #(N,1,1,1)
        out = self.sigmoid(out) #output value between [0,1]
        out = out.view(-1)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()
    

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.conditionExpand = nn.Sequential(
            nn.Linear(24, c_dim),
            nn.ReLU()
        )
        kernel_size = (4, 4)
        channels = [z_dim + c_dim, 512, 256, 128, 64]
        paddings = [(0,0), (1,1), (1,1), (1,1)]
        for i in range(1, len(channels)):
            setattr(
                self, 'convT'+str(i), nn.Sequential(
                    nn.ConvTranspose2d(channels[i-1], channels[i], kernel_size, stride = (2,2), padding = paddings[i-1]),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU()
                )
            )
        self.convT5 = nn.ConvTranspose2d(channels[-1], 3, kernel_size, stride = (2,2), padding = (1,1))
        self.tanh = nn.Tanh()

    def forward(self, z, c):
        """
        z: (B,100)
        c: (B,24)
        return: (B,3,64,64)
        """
        z = z.view(-1, self.z_dim, 1, 1)
        c = self.conditionExpand(c).view(-1, self.c_dim, 1, 1)
        out = torch.cat((z,c), dim = 1) # (N, z_dim+c_dim,1,1)
        out = self.convT1(out) # (N,512,4,4)
        out = self.convT2(out) # (N,256,8,8)
        out = self.convT3(out) # (N,128,16,16)
        out = self.convT4(out) # (N,64,32,32)
        out = self.convT5(out) # (N,3,64,64)
        out = self.tanh(out) #output values between [-1,+1]
        return out
    
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()
    



