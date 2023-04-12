# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:04:39 2023

@author: SHREYASH
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
import wandb
import argparse
#--------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------Dataset creation and augmentation------------------------------------
#--------------------------------------------------------------------------------------------------------------------
def dataset_creation(data_augmentation=True):
  if data_augmentation:
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ])
  else:
        train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ])
  
  val_transforms = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

  ])

  #CHANGE THIS TO YOUR DIRECTORY OF STORED DATASET
  data_dir = r'C:\Users\SHREYASH\Desktop\CS6910\Assignment 2\Dataset\nature_12K\inaturalist_12K\train'
  data_dir_2 = r'C:\Users\SHREYASH\Desktop\CS6910\Assignment 2\Dataset\nature_12K\inaturalist_12K\val'
  # Create training and validation datasets
  train_dataset = ImageFolder(data_dir, transform=train_transforms)
  train_size = int(0.8 * len(train_dataset))
  val_size = len(train_dataset) - train_size
  train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
  val_dataset = ImageFolder(data_dir, transform=val_transforms)
  test_dataset = ImageFolder(data_dir_2, transform=val_transforms)
   
  # Create training and validation data loaders
  #Need to add this in the main function and make batch_size hyperparameter from 32 to 64
  BATCH_SIZE=32
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
  return train_loader,val_loader

train_loader,val_loader=dataset_creation()

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------Creating model---------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
class MyModel(nn.Module):
    def __init__(self, num_filter=[16,16,16,16,16], kernel_size=[5, 5, 5, 5, 5], stride=[1, 1, 1, 1, 1], padding=[1, 1, 1, 1, 1], activation='relu',batch_normalization='False',dropout=0.2):
        super(MyModel, self).__init__()
        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.batch_normalization=batch_normalization

        self.conv1 = nn.Conv2d(3, num_filter[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(num_filter[0])
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(num_filter[0], num_filter[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(num_filter[1])
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(num_filter[1], num_filter[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(num_filter[2])
        self.dropout3 = nn.Dropout(dropout)
        self.conv4 = nn.Conv2d(num_filter[2], num_filter[3], kernel_size=kernel_size[3], stride=stride[3], padding=padding[3])
        self.bn4 = nn.BatchNorm2d(num_filter[3])
        self.dropout4 = nn.Dropout(dropout)
        self.conv5 = nn.Conv2d(num_filter[3], num_filter[4], kernel_size=kernel_size[4], stride=stride[4], padding=padding[4])
        self.bn5 = nn.BatchNorm2d(num_filter[4])
        self.dropout5 = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        
        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dense=nn.Linear(num_filter[4],256)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_normalization: x = self.bn1(x)         
        x = self.activation(x)
        x=self.dropout1(x)
        x = self.pooling(x)
      
        x = self.conv2(x)
        if self.batch_normalization: x = self.bn2(x)
        x = self.activation(x)
        x=self.dropout2(x)
        x = self.pooling(x)

        x = self.conv3(x)
        if self.batch_normalization: x = self.bn3(x)
        x = self.activation(x)
        x=self.dropout3(x)
        x = self.pooling(x)

        x = self.conv4(x)
        if self.batch_normalization: x = self.bn4(x)
        x = self.activation(x)
        x=self.dropout4(x)
        x = self.pooling(x)

        x = self.conv5(x)
        if self.batch_normalization: x = self.bn5(x)
        x = self.activation(x)
        x=self.dropout5(x)
        x = self.pooling(x)

        x = self.avgpool(x)#read why cant we use maxpooling here
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = self.fc(x)

        return x
    


#-------------------------------------------------------------------------------------------------------------------
#------------------------------------running everything-------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
      print('Using GPU...!')
    else:
      print('Using CPU...!(terminate the runtime and restart using GPU)')
      
      
    parser = argparse.ArgumentParser(description='Train a CNN on iNaturalist Dataset...')
    parser.add_argument('--wandb_entity', type=str, default='shreyashgadgil007', help='Name of the wandb entity')
    parser.add_argument('--wandb_project', type=str, default='CS6910-Assignment2', help='Name of the wandb project')
    parser.add_argument('--activation', type=str, default='elu', help='choices:[relu,gelu,elu,silu]')
    parser.add_argument('--num_filter', type=list, default=[64,128,256,512,1024], help='Enter 5 filters list')
    parser.add_argument('--kernel_size', type=list, default=[5,5,5,5,5], help='Enter 5 kernel values')
    parser.add_argument('--dropout', type=int, default=0.4, help='choices:(0,1)')
    parser.add_argument('--batch_norm', type=bool, default=False, help='choices:["True","False"]')
    parser.add_argument('--data_augmentation', type=bool, default=False, help='choices:["True","False"]')
      
    args = parser.parse_args()
        
    sweep_config = {
        'method': 'bayes', #grid, random,bayes
        'metric': {
          'name': 'val_accuracy',
          'goal': 'maximize'  
        },
        'parameters': {
            'activation': {
                'values': [args.activation]
            },
            'num_filter': {
                'values': [args.num_filter]
            },        
            'kernel_size':{
                'values':[args.kernel_size]
            },
            'dropout':{
                'values':[args.dropout]
            },
            'batch_norm':{
                'values':[args.batch_norm]
            },
            'data_augmentation':{
                'values':[args.data_augmentation]
            },
            
        }
    }
    def sweep_train():
        # Default values for hyper-parameters we're going to sweep over
        config_defaults = {
            'activation':'relu',
            'num_filter':[256, 256, 256, 256, 256],
            'kernel_size':[3,3,3,3,3],
            'dropout':0.2,
            'batch_norm':True,
            'data_augmentation':True,
        }

        # Initialize a new wandb run
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,config=config_defaults)
        wandb.run.name = 'EVALUATION RUN(ED22S016):'+'act:'+ str(wandb.config.activation)+' ;filter:'+str(wandb.config.num_filter)+ ' ;ker:'+str(wandb.config.kernel_size)+ ' ;drop:'+str(wandb.config.dropout)+' ;b_n:'+str(wandb.config.batch_norm)+' ;d_a:'+str(wandb.config.data_augmentation)

        
        config = wandb.config
        activation = config.activation
        num_filter = config.num_filter
        kernel_size = config.kernel_size
        dropout = config.dropout
        batch_norm = config.batch_norm
        data_augmentation = config.data_augmentation
        # Model training here
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyModel( num_filter=num_filter,kernel_size=kernel_size, dropout=dropout, activation=activation,batch_normalization=batch_norm).to(device)
        train_loader,val_loader=dataset_creation(data_augmentation)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.0003, betas=(0.9,0.9999))
        # Train the model
        num_epochs = 15
        for epoch in range(num_epochs):
                # Set to training mode
                model.train()

                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                train_epoch_loss = running_loss / len(train_loader.dataset)
                train_epoch_acc = (running_corrects.double() / len(train_loader.dataset))*100

                # Set to evaluation mode
                model.eval()

                running_loss = 0.0
                running_corrects = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        # Forward pass
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                val_epoch_loss = running_loss / len(val_loader.dataset)
                val_epoch_acc = (running_corrects.double() / len(val_loader.dataset))*100
                
                print(f"Epoch {epoch+1}/{num_epochs}--> Training_Loss:{train_epoch_loss:.2f}; Train_Accuracy:{train_epoch_acc:.2f}; Validation_Loss:{val_epoch_loss:.2f}; Val_Accuracy:{val_epoch_acc:.2f}")
                wandb.log({"train_loss":train_epoch_loss,"train_accuracy": train_epoch_acc,"val_loss":val_epoch_loss,"val_accuracy":val_epoch_acc},)
                #emptying the cache after one complete run
                if epoch==num_epochs-1:
                    torch.cuda.empty_cache()


    sweep_id = wandb.sweep(sweep_config, entity=args.wandb_entity, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train, count=120)

