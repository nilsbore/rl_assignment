'''
In this assignment, we consider an appliction of neural fitted Q-learning to a navigation problem on a grid. Our agent can move in 4 directions: up, down, left, right. The goal is a randomly picked cell of the grid. Instead of directly perceiving its position, the agent gets a 10-dimensional observation vector from the environment. Note that this way we deal with a continuous state space. Therefore, the goal of the assignment is to implement neural fitted Q-learning.

Hint: use PCA to reduce the dimensionality of the observations.
'''


from scipy import zeros, clip, asarray
from enum import IntEnum
import random
import numpy as np

from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments import EpisodicTask
from pybrain.rl.learners.valuebased import ActionValueNetwork, ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import NFQ, Q
from pybrain.rl.experiments import EpisodicExperiment, Experiment
from pybrain.rl.explorers import EpsilonGreedyExplorer 

from sklearn.decomposition import PCA


class Action(IntEnum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
'''
This class specifies the behaviour of the environment (the grid).  
'''
class NavigationEnv(Environment):
    def __init__(self, height, width, measurement_model):   
        self.width = width
        self.height = height
        self.position = (int(self.width/2), int(self.height/2))
        self.measurement_model = measurement_model

    
    def getSensors(self):
        #you can modify this method if you wish to reduce the dimensionality of the observations
        """ 
        the currently visible state of the world 
            
        """ 
        observation = np.dot(self.measurement_model, np.array(self.position))
        observation_after_reduction = pca.transform([observation,])[0][0:2]         
        return  observation_after_reduction
        
        
    
    def getPosition(self):
        return  np.array(self.position)                
                    
                    
    def performAction(self, action):
        """ perform an action on the world that changes it's internal state (makes a move)
            
        """
        #moving the agent with respect to the action   
        if action+1 == Action.UP:
            self.position = (self.position[0], self.position[1] - 1)
        elif action+1 == Action.DOWN:
            self.position = (self.position[0], self.position[1] + 1)
        elif action+1 == Action.LEFT:
            self.position = (self.position[0] - 1, self.position[1])
        elif action+1 == Action.RIGHT:
            self.position = (self.position[0] + 1, self.position[1])
   
        #if the agent has reached the boundary of the grid, it cannot move further    
        if self.position[0] < 0:
            self.position = (0, self.position[1])
        elif self.position[1] < 0:
            self.position = (self.position[0], 0)
        elif self.position[0] >= self.width:
            self.position = (self.width-1, self.position[1])
        elif self.position[1] >= self.height:
            self.position = (self.position[0], self.height-1)
                 
    def reset(self):
        self.position = (int(self.width/2), int(self.height/2))


'''
This class establishes the relationship between the agent and the environment -- i.e., specifies the task.  
'''
class NavigationTask(EpisodicTask):
    def __init__(self, environment, goal):
        self.env = environment
        self.goal = goal
        self.t = 0
        self.N = 5000

    def performAction(self, action):
    #updates the timer and changes the position of the agent with respect to the action
        self.t += 1       
        self.env.performAction(action)
        
    def getObservation(self):
    #gets an observation from the environment
        sensors = self.env.getSensors()
        return sensors
    
    def getReward(self):
    #gives 10000 points if the goal is reached and -1 otherwise
        position = self.env.getPosition() 
        if position[0] == self.goal[0] and position[1] == self.goal[1]:
            reward = 10000
        else:
            reward = -1
        return reward
 
        
    def reset(self):    
    #resets the timer and puts the agent to the initial position
        self.env.reset()
        self.t = 0 
        
        
    def isFinished(self):
    #this is method should return True when the episode (a walk in the grid) is finished -- i.e., when the goal is reached or the time is out

        position = self.env.getPosition()
        if position[0] == self.goal[0] and position[1] == self.goal[1]:
            return True
        if self.t >= self.N:
            return True
            
        return False

        
#----------------------------------------------------------------------------------        
'''
Here we initialize oor problem and create the environment
'''
#set the size of the grid
width = 10
height = 10
#set the dimensionality of the distorted observations
measurement_dimension = 10
#the distortion model -- a random vector
measurement_model = np.random.rand(measurement_dimension, 2)
#create the environment
env = NavigationEnv(height, width, measurement_model)
#set the goal
goal = (random.randint(0, width-1), random.randint(0, height-1))
#initialize the task
task = NavigationTask(env, goal) 


'''
This is what the student is supposed to do
'''
#PCA
points = []
measurements = []
#Here we generate 10000 random positions, multiply each of them by the distortion vector and use for learning
for n in range(1, 10000):
     position_sample = (random.randint(0, width-1), random.randint(0, height-1))
     observation = np.dot(measurement_model, np.array(position_sample))
     points.append(position_sample)
     measurements.append(observation)

pca = PCA()
pca.fit(measurements)

#Learning
module = ActionValueNetwork(2, 4)
learner = NFQ()
learner._setExplorer(EpsilonGreedyExplorer(0.9))
agent = LearningAgent(module, learner)
experiment = EpisodicExperiment(task, agent)


for n in range(1, 150):
     print "episode #", n
     if n <= 120:
        learner._setExplorer(EpsilonGreedyExplorer(0.9))
        print "learning: epsilon = 0.9"

     elif n > 120 and n <= 150:
        learner._setExplorer(EpsilonGreedyExplorer(0.1))
        print "testing: epsilon = 0.1"
     
     print "total reward of the episode:"
     print np.sum(experiment.doEpisodes(1))
     agent.learn(1)

