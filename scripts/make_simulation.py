#!/usr/bin/python3

from enum import Enum
import random
#import cv2
import numpy as np

class Action(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Game(object):

    def __init__(self, height, width, goal, measurement_model):

        self.width = width
        self.height = height
        self.goal = goal
        self.position = (int(self.width/2), int(self.height/2))
        self.measurement_model = measurement_model

    def step(self, action):

        print(action)
        if action == Action.UP:
            self.position = (self.position[0], self.position[1] - 1)
        elif action == Action.DOWN:
            self.position = (self.position[0], self.position[1] + 1)
        elif action == Action.LEFT:
            self.position = (self.position[0] - 1, self.position[1])
        elif action == Action.RIGHT:
            self.position = (self.position[0] + 1, self.position[1])
        else:
            print("This is not a valid action")

        if self.position[0] < 0:
            self.position = (0, self.position[1])
        elif self.position[1] < 0:
            self.position = (self.position[0], 0)
        elif self.position[0] >= self.width:
            self.position = (self.width-1, self.position[1])
        elif self.position[1] >= self.height:
            self.position = (self.position[0], self.height-1)

        if self.position[0] == self.goal[0] and self.position[1] == self.goal[1]:
            reward = 1000
        else:
            reward = -1

        return reward

    def measure(self):

        return self.measurement_model*np.array(self.position)

    def display(self):

        scene = np.zeros((self.height, self.width), dtype=int)
        scene[self.position[1], self.position[0]] = 1
        print(2*self.width*"=")
        print(scene)
        

def run():
 
    width = 20
    height = 20
    goal = (random.randint(0, width-1), random.randint(0, height-1))
    measurement_dimension = 10
    measurement_model = np.random.rand(measurement_dimension, 2)
    epsilon = 0.05
    
    for n in range(0, 1):
        game = Game(height, width, goal, measurement_model)
        total_reward = 0
        # just do a random walk
        for i in range(0, 1000):
            u = 0. #random.rand()
            if u > epsilon:
                pass # here, we do steps with the actual policy
            else:
                action = Action(random.randint(1, 4))
                reward = game.step(action)
            measurement = game.measure()
            total_reward += reward
            game.display()
            if reward == 1000:
                break
        print(total_reward)

        # here, we should update our estimates of the q-values

if __name__ == '__main__':
    run()


