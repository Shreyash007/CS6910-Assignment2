# CS6910: Assignment-2
## Problem Statement
Learn how to use CNNs: train from scratch and finetune a pre-trained model as it is.

## Prerequisites

```
python 3.9
numpy 1.21.5
pytorch
wget
```
 - Clone/download  this repository
 - I have conducted all my experiments in Google Collab, for running in google colab, install wandb and wget(for importing dataset) using following command 
 - Enable GPU on colab for faster training
 
  ``` 
  !pip install wandb 
  !pip install wget
  !pip install timm(for EfficinetNetV2)
  ```
 - For running locally, install wandb and other required libraries using following command  
  ``` 
  pip install wandb
  pip install numpy
  pip install pytorch
  pip install timm(for EfficinetNetV2)
  ```

## Dataset
- We use [iNaturalist](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) dataset for our experiments.

# Part A
### Hyperparameters used in experiments for Part A
|Sr. no| Hyperparameter| Variation/values used|
|------|---------------|-----------------|
|1.| Activation function| ReLu,GeLU,SiLu, Elu|
|2.| Num_filters| [256,256,256,256,256],[128,128,128,128,128],[64,128,256,512,1024],[64,64, 64,64,64]|
|3.| Kernel size| [3,3,3,5,5],[5,5,5,5,5],[3,3,3,3,3]|
|4.| Drop_out| 0.2,0.4 |
|5.| Batch_norm| True, False |
|6.| Data augmentation| True, False |

- Note: I have used ADAM optimizer with 0.0003 learning rate and beta1=0.93 for the above experiments

###  Code for Part A

The experiments for Part A can be found [here](https://github.com/Shreyash007/CS6910-Assignment2/blob/main/Assignment2_part-A.ipynb).

# Part B
### Hyperparameters used in experiments for Part B
|Sr. no| Hyperparameter| Variation/values used|
|------|---------------|-----------------|
|1.| Freeze percent| 0.25,0.5,0.75|
|2.| Learning rate| 0.0001,0.0003|
|3.| Beta1| 0.9,0.93,0.95|

### Code for Part B

The experiments on Part B can be found [here](https://github.com/Shreyash007/CS6910-Assignment2/blob/main/Assignment2_part-B.ipynb).


## Evaluation file for part A(PartA_cmd.py)

For evaluating model download [PartA_cmd.py](https://github.com/Shreyash007/CS6910-Assignment2/blob/main/PartA_cmd.py) file. (make sure you have all the prerequisite libraries installed). 
Make sure you have downloaded the dataset and the directory of dataset should be changed in line 53,54
```data_dir = r'C:\Users\SHREYASH\Desktop\CS6910\Assignment 2\Dataset\nature_12K\inaturalist_12K\train'``` 
```data_dir_2 = r'C:\Users\SHREYASH\Desktop\CS6910\Assignment 2\Dataset\nature_12K\inaturalist_12K\val'```
To the directory location of your stored dataset

And run the following command in the command line(this will take the default arguments).
```
python PartA_cmd.py 
```
The default evaluation run can be seen [here](https://wandb.ai/shreyashgadgil007/shreyashgadgil007/runs/) in wandb.


The arguments supported by PartA_cmd.py file are:

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `--wandb_project` | "CS6910-Assignment2" | Project name used to track experiments in Weights & Biases dashboard |
| `--wandb_entity` | "shreyashgadgil007"  | Wandb Entity used to track experiments in the Weights & Biases dashboard. | 
| `--num_filter` |[64,128,256,512,1024] | Enter 5 filters list | 
| `--activation` | 'elu' | choices:[relu,gelu,elu,silu] |
| `--kernel_size` | [5,5,5,5,5] | Enter 5 kernel values |
| `--dropout_rate` | 0.4 | choice in range (0,1) |
| `--batch_norm` | False | True/False |
| `--data_augmentation` | False | True/False |

Supported arguments can also be found by:
```
python PartA_cmd.py -h
```
## Evaluation file for part B([PartB_cmd.py](https://github.com/Shreyash007/CS6910-Assignment2/blob/main/PartB_cmd.py))
Please follow the instructions of Part A, for loading dataset.
And run the following command in the command line(this will take the default arguments).
```
python PartB_cmd.py 
```
The default evaluation run can be seen [here](https://wandb.ai/shreyashgadgil007/shreyashgadgil007/runs/) in wandb.


The arguments supported by PartB_cmd.py file are:

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `--wandb_project` | "CS6910-Assignment2" | Project name used to track experiments in Weights & Biases dashboard |
| `--wandb_entity` | "shreyashgadgil007"  | Wandb Entity used to track experiments in the Weights & Biases dashboard. | 
| `--freeze_percent` | 0.25 | choice in range (0,1) |
| `--learning_rate` | 0.0003 | choice in range (0,1)|
| `--beta1` | 0.93 | choice in range (0,1) |

## Report

The wandb report for this assignment can be found [here](https://wandb.ai/shreyashgadgil007/CS6910-Assignment2/reports/CS6910-Assignment-2--VmlldzozOTAzMTc4).
## Author
[Shreyash Gadgil](https://github.com/Shreyash007)
ED22S016
