import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        #add epsilon
        self.epsilon = 0.005
        #add alpha
        self.alpha = 1.0
        #add gamma
        self.gamma = 1.0




    def probabilities(self, state):
        """ Obtains the probabilities of each action given a state.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - probs: a list of probabilities of each action given the state
        """
        probs = [1-self.epsilon+self.epsilon/self.nA if a == np.argmax(self.Q[state])\
                 else self.epsilon/self.nA\
                 for a in np.arange(self.nA) ]
        return probs




    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """



        if all(self.Q[state][a] == 0 for a in range(self.nA)):
            action = np.random.choice(self.nA)
        else:
            action = np.random.choice(np.arange(self.nA), p= self.probabilities(state))

        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if done:
            self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + self.gamma * 0 - self.Q[state][action]))
        else:
            self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma*np.dot(self.Q[next_state],self.probabilities(next_state)) -self.Q[state][action])

