import numpy as np
import gym
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import pickle
import logging

"""
This codebase consists of the code to train 
our agent in the LunarLander environment. 
This codebase uses a Deep Q-Learning algorithm
with a technique calld experience replay. 

"""

def deepQNetwork(alpha,action_space_dims,state_space_dims,layer1_dims,layer2_dims):
    """
    This function builds the 2 layer feed forward network using Keras.

    Args:
        alpha : Learning rate
        action_space_dims : size of the action space
        state_space_dims :  size of the state space vector
        layer1_dims: size of layer 1
        layer2_dims : size of layer 2

    Returns:
        [type]: [description]
    """
    DQN=Sequential([ Dense(layer1_dims, input_shape=(state_space_dims,)),
                      Activation('relu'), Dense(layer2_dims),Activation('relu'),
                      Dense(action_space_dims)])

    DQN.compile(optimizer=Adam(lr=alpha), loss='mse')

    return DQN

class ReplayBuffer(object):
    def __init__(self,memory_limit,state_space_dims,action_space_dims):
        # set max memory limit
        self.memory_limit=memory_limit  

        # set input state shape dimensions
        self.input_shape=state_space_dims

        # initialize array to store current state 
        self.state_experiences=np.zeros((self.memory_limit, state_space_dims))

        # initialize array to store actions 
        self.action_experiences=np.zeros((self.memory_limit,action_space_dims), dtype=np.int8)

        # initialize array to store rewards
        self.reward_experiences=np.zeros(self.memory_limit)

        # initialize array to store new states
        self.new_state_experiences=np.zeros((self.memory_limit, state_space_dims))       
        self.terminal_memory=np.zeros(self.memory_limit,dtype=np.float32)
        self.memory_counter=0
  

    def store_experience(self,experience_tuple):

        """
        This function stores the experience tuple into the replay memory.
        Experience tuple is of the form (state,action,reward,new_state,done)

        """
        # get index
        index=self.memory_counter % self.memory_limit

        # store current state 
        self.state_experiences[index]=experience_tuple[0]
        self.terminal_memory[index]=1-int(experience_tuple[4]) 

        # store action taken        
        actions=np.zeros(self.action_experiences.shape[1])
        actions[experience_tuple[1]]=1.0
        self.action_experiences[index]=actions

        # store reward received 
        self.reward_experiences[index]=experience_tuple[2]

        # store new state 
        self.new_state_experiences[index]=experience_tuple[3]
        self.memory_counter+=1
    
    def extract_batch(self,batch_size):
        """
        This function extracts a batch from the Replay Memory
        to be processed by the neural network.

        Args:
            batch_size : No. of samples to be extracted

        Returns:
            experience batch : 
        """

        # get size of replay memory
        max_memory=min(self.memory_counter, self.memory_limit)        
        batch=np.random.choice(max_memory,batch_size)
        experience_batch=[self.state_experiences[batch], self.action_experiences[batch], self.reward_experiences[batch],
                          self.new_state_experiences[batch], self.terminal_memory[batch]]

        return experience_batch

class Agent(object):
    def __init__(self,alpha=0.0005,gamma=0.99,n_actions=4,epsilon=1,batch_size=64, input_dims=8,epsilon_dec=0.996,epsilon_end=0.01,
                 mem_size=1000000,layer1_dims=256,layer2_dims=256, model_name='lunarLander.h5'):

        """
        Initialize the agen. All the hyperparameters
        such as the learning rate, discount factor,
        exploration/exploitation parameter, layer 
        dims are initialized here """
        
        self.action_space=[i for i in range(n_actions)]
        self.n_actions=n_actions
        self.gamma=gamma
        self.epsilon=epsilon
        self.epsilon_dec=epsilon_dec
        self.epsilon_min=epsilon_end
        self.batch_size=batch_size
        self.model_file=model_name
        self.memory=ReplayBuffer(mem_size,input_dims,n_actions)
        self.model=deepQNetwork(alpha, n_actions, input_dims, layer1_dims,layer2_dims)


    
    def choose_action(self,state):
        """
        This function chooses the next step for 
        agent based on the exploration-exploitation
        parameter epsilon

        Args:
            state: Current state of the agent

        Returns:
            Action: The action which the agent should take
        """
        state=state[np.newaxis,:]
        # get random float between (0,1)
        random_float=np.random.random()
        # exploit if random float is less than epsilon
        if random_float > self.epsilon:
            actions=self.model.predict(state)
            action=np.argmax(actions)  
        # explore if random float is greater than epsilon          
        else:
            action=np.random.choice(self.action_space)            
        return action
    
    
    def save_model(self):
        """
        Function the save the model
        """
        self.model.save(self.model_file)

    def load_model(self):
        """
        Function to load the model
        """
        self.model=load_model(self.model_file)
        
    def remember(self,experience_tuple):
        """
        Function to store the experience tuple 
        in the replay buffer

        Args:
            experience_tuple: tuple consiting of the agent's experience
        """
        self.memory.store_experience(experience_tuple)

    def train(self):

        # return and end training if not eneough expriences in replay buffer
        if self.memory.memory_counter < self.batch_size:            
            return

        # extract training batch
        state,action,reward,new_state,done = self.memory.extract_batch(self.batch_size)
        action_values=np.array(self.action_space,dtype=np.int8)
        action_indices=np.dot(action, action_values)

        # update value of epsilon 
        if self.epsilon> self.epsilon_min:
            self.epsilon=self.epsilon*self.epsilon_dec
        else:
            self.epsilon=self.epsilon_min   

        # evaluate q_value for the current state,action pair
        current_q=self.model.predict(state)

        #evaluate q_value for next state,action pair
        next_q=self.model.predict(new_state)        
        q_target=current_q.copy()
        batch_index=np.arange(self.batch_size,dtype=np.int32)

        # find target q value using bellmans equation
        q_target[batch_index,action_indices]= reward + self.gamma*np.max(next_q,axis=1)*done
        _=self.model.fit(state,q_target,verbose=0)

    


def main():

    """
    This the main training function.
    """

    env=gym.make('LunarLander-v2')
    #NN_dimensions=[[256,256],[256,128],[128,128],[512,256]]
        
    episodes=600   

    # Initialize the Logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("Main")
    logger.setLevel(logging.INFO)  

    # initialize the agent
    agent=Agent()
    scores=[]
    mean_scores=[]   

    # iterate over number of episodes
    for i in range(episodes+1):
        done = False
        score = 0
        # rest the environment
        current_state=env.reset()        

        while not done:
            # choose action based on either exploration/exploitation
            action=agent.choose_action(current_state)
            # take that action and store next state
            new_state,reward,done,_= env.step(action)
            score+=reward
            # store experience
            experience_tuple=(current_state,action,reward,new_state,done)
            agent.remember(experience_tuple)
            current_state=new_state
            # train the model and update weights
            agent.train()
        #append to scores list
        scores.append(score)
        # find the average score
        avg_score=np.mean(scores)
        # store the average score
        mean_scores.append(avg_score)

        # print log
        logger.info(" Episode no. : {:.2f} Score : {:.2f} Average Score : {:.2f}".format(i,score,avg_score))

    # save model                   
    agent.save_model()
    
    # dump output, used for plotting 
    filename='scores.txt'
    with open(filename, 'wb') as f:
        pickle.dump(scores, f) 

    filename='MeanScores.txt'
    with open(filename, 'wb') as f:
        pickle.dump(mean_scores, f) 

if __name__ == '__main__' :
    main()





    










