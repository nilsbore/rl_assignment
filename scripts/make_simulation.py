#!/usr/bin/python3

from enum import IntEnum
import random
import numpy as np

class Action(IntEnum):
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

        return np.dot(self.measurement_model, np.array(self.position))

    def display(self, display=False):

        if display:
            scene = np.zeros((self.height, self.width), dtype=int)
            scene[self.position[1], self.position[0]] = 1
            print(2*self.width*"=")
            print(scene)
        

class LinearRegressor(object):

    def __init__(self, measurement_dimension):

        self.p = np.zeros((4, measurement_dimension+1))
    
    def fit(self, measurements, actions, total_rewards):

        for i in range(1, 5):
            inds = np.where(actions == i)[0]
            N = len(inds)
            A = np.hstack((measurements[inds], np.ones((N, 1))))
            b = total_rewards[inds]
            self.p[i-1] = np.linalg.lstsq(A, b)[0]

    def predict(self, measurement):

        predicted_rewards = np.dot(self.p, np.append(measurement, 1.0))
        return predicted_rewards

def run():
 
    width = 20
    height = 20
    goal = (random.randint(0, width-1), random.randint(0, height-1))
    measurement_dimension = 10
    measurement_model = np.random.rand(measurement_dimension, 2)
    epsilon = 0.8

    measurements = None # this will contain all of our measurements
    actions = None # this will contain all of our actions
    total_rewards = None # this will contain all of our future rewards
    
    regressor = LinearRegressor(measurement_dimension)
    reward_history = []
    for n in range(0, 100):
        game = Game(height, width, goal, measurement_model)
        
        run_measurements = []
        run_actions = []
        total_reward = 0
        # just do a random walk
        for i in range(0, 1000):
            u = random.random()
            measurement = game.measure()
            if u > epsilon and n > 5:
                predicted_rewards = regressor.predict(measurement)
                action = Action(np.argmax(predicted_rewards)+1)
            else:
                action = Action(random.randint(1, 4))
            reward = game.step(action)
            total_reward += reward
            game.display(display=False)

            run_actions.append(int(action))
            run_measurements.append(measurement)

            if reward == 1000:
                break

        episode_len = len(run_measurements)

        if measurements is None:
            measurements = np.vstack(run_measurements)
            actions = np.array(run_actions, dtype=int)
            total_rewards = total_reward*np.ones((episode_len,))
        else:
            measurements = np.concatenate((measurements, np.vstack(run_measurements)), 0)
            actions = np.concatenate((actions, np.array(run_actions, dtype=int)), 0)
            total_rewards = np.concatenate((total_rewards, total_reward*np.ones((episode_len,))), 0)

        reward_history.append(total_reward)

        # here, we should update our estimates of the q-values
        regressor.fit(measurements, actions, total_rewards)

    print("Reward history:")
    print(reward_history)
    print("Mean reward:")
    print(np.mean(reward_history))

if __name__ == '__main__':
    run()


