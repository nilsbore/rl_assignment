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


class Action(IntEnum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class NavigationEnv(Environment):

    # the number of action values the environment accepts
    #indim = 4
    
    # the number of sensor values the environment produces
    #outdim = 1
    
    def __init__(self, height, width, measurement_model):
        self.width = width
        self.height = height
        #self.position = (int(self.width/2), int(self.height/2))
        self.position = (9, 9)
        self.measurement_model = measurement_model
        print "env initialized"
    
    def getSensors(self):
        """ the currently visible state of the world (the    observation may be stochastic - repeated calls returning different values) 
            :rtype: by default, this is assumed to be a numpy array of doubles
        """
        #hand_value = int(raw_input("Enter hand value: ")) - 1
        #return [float(hand_value),]
       # np.dot(self.measurement_model, np.array(self.position)) != None
        
        #return np.dot(self.measurement_model, np.array(self.position))
        return np.array(self.position)
        
    
    def getPosition(self):
        """ the currently visible state of the world (the    observation may be stochastic - repeated calls returning different values) 
            :rtype: by default, this is assumed to be a numpy array of doubles
        """
        #hand_value = int(raw_input("Enter hand value: ")) - 1
        #return [float(hand_value),]
        return  np.array(self.position)                
                    
    def performAction(self, action):
        """ perform an action on the world that changes it's internal state (maybe stochastically).
            :key action: an action that should be executed in the Environment. 
            :type action: by default, this is assumed to be a numpy array of doubles
        """   
        if action+1 == Action.UP:
            self.position = (self.position[0], self.position[1] - 1)
            #print "UP"
        elif action+1 == Action.DOWN:
            self.position = (self.position[0], self.position[1] + 1)
            #print "DOWN"
        elif action+1 == Action.LEFT:
            self.position = (self.position[0] - 1, self.position[1])
            #print "LEFT"
        elif action+1 == Action.RIGHT:
            self.position = (self.position[0] + 1, self.position[1])
            #print "RIGHT"
   
            
        if self.position[0] < 0:
            self.position = (0, self.position[1])
        elif self.position[1] < 0:
            self.position = (self.position[0], 0)
        elif self.position[0] >= self.width:
            self.position = (self.width-1, self.position[1])
        elif self.position[1] >= self.height:
            self.position = (self.position[0], self.height-1)
            
        #print self.position
        #print self.position
        #print "Action performed: ", action
     
    def reset(self):
        #self.position = (int(self.width/2), int(self.height/2))
        self.position = (9, 9)

class NavigationTask(EpisodicTask):

    def __init__(self, environment, goal):
        self.env = environment
        # we will store the last reward given, remember that "r" in the Q learning formula is the one from the last interaction, not the one given for the current interaction!
        self.lastreward = 0
        self.goal = goal
        self.t = 0
        self.N = 5000

    def performAction(self, action):
        self.t += 1
        
        self.env.performAction(action)
        
    def getObservation(self):
        sensors = self.env.getSensors()
        #print sensors
        return sensors
    
    def getReward(self):
        #reward = raw_input("Enter reward: ")
        position = self.env.getPosition()
 
       # if position[0] == self.goal[0] and position[1] == self.goal[1]:
        if position[0] < 1 and position[1] < 1:
            reward = 10000
        else:
            reward = -1
 
        # retrieve last reward, and save current given reward
        #cur_reward = self.lastreward
        #self.lastreward = reward
    
        #return cur_reward
        #print reward
        return reward
        
    def reset(self):
        self.env.reset()
        self.t = 0 
        
    def isFinished(self):
        position = self.env.getPosition()
        #if position[0] == self.goal[0] and position[1] == self.goal[1]:
        #if position[0] < 5 and position[1] < 5:
        if position[0] < 1 and position[1] < 1:
            return True
        if self.t >= self.N:
            return True
        return False
        
    @property
    def indim(self):
        return self.env.indim
    
    @property
    def outdim(self):
        return self.env.outdim
        
#----------------------------------------------------------------------------------        


module = ActionValueNetwork(2, 4)
#module = ActionValueTable(100, 4)
learner = NFQ()
#learner = Q()
learner._setExplorer(EpsilonGreedyExplorer(0.9))
agent = LearningAgent(module, learner)



 
width = 10
height = 10
#goal = (random.randint(0, width-1), random.randint(0, height-1))
goal = (0, 0)
measurement_dimension = 2
#measurement_model = np.ones((2,2))
measurement_model = np.random.rand(measurement_dimension, 2)
env = NavigationEnv(height, width, measurement_model)
task = NavigationTask(env, goal)
experiment = EpisodicExperiment(task, agent)



    
for n in range(1, 150):
     if n <= 120:
        learner._setExplorer(EpsilonGreedyExplorer(0.8))
        print "epsilon = 0.8"

     elif n > 120 and n <= 150:
        learner._setExplorer(EpsilonGreedyExplorer(0.1))
        print "epsilon =  0.1"

     print np.mean([sum(x) for x in experiment.doEpisodes(1)])
     agent.learn(1)
 
