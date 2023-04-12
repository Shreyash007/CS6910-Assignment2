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
  ```
 - For running locally, install wandb and other required libraries using following command  
  ``` 
  pip install wandb
  pip install numpy
  pip install pytorch
  pip install wget
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

The code for Part A can be found [here](https://github.com/Shreyash007/CS6910-Deep-Learning-Course/blob/main/Assignment1(Q1_Q3).ipynb).

# Part B
### Hyperparameters used in experiments for Part B
|Sr. no| Hyperparameter| Variation/values used|
|------|---------------|-----------------|
|1.| Freeze percent| 0.25,0.5,0.75|
|2.| Learning rate| 0.0001,0.0003|
|3.| Beta1| 0.9,0.93,0.95|

### Code for Part B

The code for Part A can be found [here](https://github.com/Shreyash007/CS6910-Deep-Learning-Course/blob/main/Assignment1(Q1_Q3).ipynb).


## Evaluation file(train.py)

For evaluating model download [train.py](https://github.com/Shreyash007/CS6910-Deep-Learning-Course/blob/main/train.py) file. (make sure you have all the prerequisite libraries installed). 


And run the following command in the command line(this will take the default arguments).
```
python train.py 
```
The default evaluation run can be seen [here](https://wandb.ai/shreyashgadgil007/shreyashgadgil007/runs/) in wandb.


The arguments supported by train.py file are:

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `--wandb_project` | "CS-6910 A1" | Project name used to track experiments in Weights & Biases dashboard |
| `--wandb_entity` | "shreyashgadgil007"  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `--dataset` | "fashion_mnist" | choices:  ["mnist", "fashion_mnist"] |
| `--epochs` | 30 |  Number of epochs to train neural network.|
| `--batch_size` | 32 | Batch size used to train neural network. | 
| `--loss_function` | "cross_entropy" | choices:  ["square_error", "cross_entropy"] |
| `--optimiser` | "nadam" | choices:  ["gd", "mgd", "ngd", "rmsprop", "adam", "nadam"] | 
| `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters | 
| `--weight_decay` | 0.0005 | Weight decay used by optimizers. |
| `--initialisation` | "xavier" | choices:  ["random", "xavier"] | 
| `--hidden_layer` | [256,256,256] | Number of hidden layers used in feedforward neural network. | 
| `--activation` | sigmoid | choices:  ["sigmoid", "tanh", "relu"] |
| `--dropout_rate` | 0.1 | choice in range (0,1) |

Supported arguments can also be found by:
```
python train.py -h
```
#### The default run has 30 epochs and  hidden layer size [256,256,256]. Hence, it may take some time to create the logs. Check the command line for runtime.

## Report

The wandb report for this assignment can be found [here](https://wandb.ai/shreyashgadgil007/CS-6910%20A1/reports/CS6910-Assignment-1--VmlldzozNTQ1MjU1).
## Author
[Shreyash Gadgil](https://github.com/Shreyash007)
ED22S016
