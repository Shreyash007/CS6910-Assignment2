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
import timm

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
def efficientnet(freeze_percent):
    # Load the EfficientNetV2 model with pre-trained weights
    model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
    
    # Freeze a percentage of layers
    count_total = sum(1 for _ in model.parameters())
    count = 0
    for param in model.parameters():
        if count < int(freeze_percent * count_total):
            param.requires_grad = False
            count += 1
        else:
            break
    
    # Replace the last layer with a new one for the specified number of classes
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 10)
    
    return model


def sweep_train():
    # Default values for hyper-parameters we're going to sweep over
        config_defaults = {
            'freeze_percent':0.25,
            'learning_rate':0.0003,
            'beta1':0.93,
        }

        # Initialize a new wandb run
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,config=config_defaults)
        wandb.run.name = 'fp:'+ str(wandb.config.freeze_percent)+' ;lr:'+str(wandb.config.learning_rate)+ ' ;beta1:'+str(wandb.config.beta1)

        
        config = wandb.config
        freeze_percent = config.freeze_percent
        learning_rate = config.learning_rate
        beta1 = config.beta1
        # Model training here
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = efficientnet(freeze_percent).to(device)
        train_loader,val_loader=dataset_creation()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=learning_rate, betas=(beta1,0.9999))#using adam
        num_epochs = 10
        for epoch in range(num_epochs):
                # Set to training mode
                model.train()

                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                train_epoch_loss = running_loss / len(train_loader.dataset)
                train_epoch_acc = (running_corrects.double() / len(train_loader.dataset))*100

                # Evaluate on validation set
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
                if epoch==num_epochs-1:
                    torch.cuda.empty_cache()
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
      
      
    parser = argparse.ArgumentParser(description='Train a CNN on iNaturalist Dataset with efficientnetV2...')
    parser.add_argument('--wandb_entity', type=str, default='shreyashgadgil007', help='Name of the wandb entity')
    parser.add_argument('--wandb_project', type=str, default='CS6910-Assignment2', help='Name of the wandb project')
    parser.add_argument('--freeze_percent', type=int, default=0.25, help='choices:range(0,1)')
    parser.add_argument('--learning_rate', type=int, default=0.0003, help='choices:range(0,1)')
    parser.add_argument('--beta1', type=int, default=0.93, help='choices:range(0,1)')
    args = parser.parse_args()
        
    sweep_config = {
    'method': 'grid', #grid, random,bayes
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'  
    },
    'parameters': {
        'freeze_percent': {
            'values': [args.freeze_percent]
        },        
        'learning_rate':{
            'values':[args.learning_rate]
        },
        'beta1':{
            'values':[args.beta1]
        }
        
    }
    }
    
    sweep_id = wandb.sweep(sweep_config, entity=args.wandb_entity, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train, count=18)



