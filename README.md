# LunarLander-v2

The main focus of this repository is to understand and implement a
Deep Q-Learning algorithm to solve Open AI Gym's Lunar Lander-v2
environment. The model was able to solve the environment,
achieving an average reward of 270 over 250 test episodes.

The code base consists of 2 files :

```
1. LunarLander.py : This file consits of the codebase used to train models. Changes can be made to various hyperparameters. 
2. Run_model.py : This file consists of code to load a pretrained model and see the output by rendering the GUI of the OpenAi LunarLander environment
```

## Requirements 
The program has been tested on Python 3.6. Also, the following modules are required to run the codebase. They can be installed using pip.`  
  
```
tensorflow==1.13.1
keras==2.2.4
numpy==1.16.2
gym==0.7.4
matplotlib==3.0.3
```

## Results 

![gif](https://user-images.githubusercontent.com/48079888/93164280-23663300-f6e7-11ea-8cf9-e049d887a4ca.gif)
