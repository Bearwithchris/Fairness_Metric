# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:27:36 2021

@author: Chris
"""

import torch
from torch import nn
from torch import optim
import torchvision
import matplotlib.pyplot as plt
#Model=====================================================================================
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer1 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=int(kwargs["input_shape"]*(2/3))
        )
        self.encoder_hidden_layer2 = nn.Linear(
            in_features=int(kwargs["input_shape"]*(2/3)), out_features=int(kwargs["input_shape"]*(1/3))
        )
        self.encoder_output_layer = nn.Linear(
            in_features=int(kwargs["input_shape"]*(1/3)), out_features=int(kwargs["input_shape"]*(1/6))
        )
        self.decoder_hidden_layer1 = nn.Linear(
            in_features=int(kwargs["input_shape"]*(1/6)), out_features=int(kwargs["input_shape"]*(1/3))
        )
        self.decoder_hidden_layer2 = nn.Linear(
            in_features=int(kwargs["input_shape"]*(1/3)), out_features=int(kwargs["input_shape"]*(2/3))
        )
        self.decoder_output_layer = nn.Linear(
            in_features=int(kwargs["input_shape"]*(2/3)), out_features=int(kwargs["input_shape"])
        )


    def forward(self, features):
        layer1 = self.encoder_hidden_layer1(features)
        activation1 = torch.relu(layer1)
        layer2 = self.encoder_hidden_layer2(activation1)
        activation2 = torch.relu(layer2)
        
        code = self.encoder_output_layer(activation2)
        code = torch.relu(code)
        
        activation3 = self.decoder_hidden_layer1(code)
        activation3 = torch.relu(activation3)
        activation4 = self.decoder_hidden_layer2(activation3)
        activation4 = torch.relu(activation4)
        
        activation4 = self.decoder_output_layer(activation4)
        reconstructed = torch.relu(activation4)
        return reconstructed
    
    def encoder(self,features):
        layer1 = self.encoder_hidden_layer1(features)
        activation1 = torch.relu(layer1)
        layer2 = self.encoder_hidden_layer2(activation1)
        activation2 = torch.relu(layer2)
        
        code = self.encoder_output_layer(activation2)
        code = torch.relu(code)
        return code
 
class AE_encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer1 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=int(kwargs["input_shape"]*(2/3))
        )
        self.encoder_hidden_layer2 = nn.Linear(
            in_features=int(kwargs["input_shape"]*(2/3)), out_features=int(kwargs["input_shape"]*(1/3))
        )
        self.encoder_output_layer = nn.Linear(
            in_features=int(kwargs["input_shape"]*(1/3)), out_features=int(kwargs["input_shape"]*(1/6))
        )
        self.decoder_hidden_layer1 = nn.Linear(
            in_features=int(kwargs["input_shape"]*(1/6)), out_features=int(kwargs["input_shape"]*(1/3))
        )
        self.decoder_hidden_layer2 = nn.Linear(
            in_features=int(kwargs["input_shape"]*(1/3)), out_features=int(kwargs["input_shape"]*(2/3))
        )
        self.decoder_output_layer = nn.Linear(
            in_features=int(kwargs["input_shape"]*(2/3)), out_features=int(kwargs["input_shape"])
        )


    def forward(self, features):
        layer1 = self.encoder_hidden_layer1(features)
        activation1 = torch.relu(layer1)
        layer2 = self.encoder_hidden_layer2(activation1)
        activation2 = torch.relu(layer2)
        
        code = self.encoder_output_layer(activation2)
        code = torch.relu(code)
        return code       
    



#Training Params======================================================================================
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
input_dim=4096*3
model = AE(input_shape=input_dim).to(device)


# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def train(model):
    #Data=========================================================
    BASE_PATH = '../data/'
    split="train"
    data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
    data=torch.flatten(data, start_dim=1)
    data=data/255.0
    trn_dataloader = torch.utils.data.DataLoader(data,batch_size=100,shuffle=False)
    
    
    
    epochs =40
    
    for epoch in range(epochs):
        loss = 0
        for batch_idx, data in enumerate(trn_dataloader):
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            data= data.to(device)
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(data)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, data)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(trn_dataloader)
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        
    torch.save(model.state_dict(),"./AE_dict_model.pt")


def load_encoder(model_path,input_dim):
    model = AE_encoder(input_shape=input_dim)
    load_dict=torch.load(model_path)
    model.load_state_dict(load_dict)
    model.eval()
    return model
    
#Debug
def deflatten(data):
    data=data.reshape(-1,3,64,64).permute(0,2,3,1).cpu().detach().numpy()
    return data
    

# train(model)