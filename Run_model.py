import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
import gym
from time import sleep

"""
This function loads in a pretrained model and
renders the LunarLander environment to show the result.
Set the model path in the model_path variable.
"""
# model path
model_path='../Models/Final_Model.h5'
env=gym.make('LunarLander-v2')
# load model
model=load_model(model_path)
# intialize number of episodes
num_episodes= 5

for i in range(num_episodes):
    done = False
    score = 0
    # reset environment
    state=env.reset()       

    while not done:
        state=state[np.newaxis,:] 
        # use model to predict action
        actions=model.predict(state)
        action=np.argmax(actions)  
        # Take that action and get new state
        new_state,reward,done,info= env.step(action)
        # display GUI
        env.render()
        sleep(0.02)
        score+=reward
        state=new_state
    env.close()




