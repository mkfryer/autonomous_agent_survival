from AgentFactory import AgentFactory
from AgentFactory import Agent
import numpy as np

class World(AgentFactory):
    """
    Objective of this class is to determine what is the well the water is
    being pumped to and create agents according to the prior type passed in.

    Parameters:
        prior_type (str): "Uniform", "Good", or "Bad"
        correctWell (str): "Well A", "Well B", or "Well C"
        wellA (x,y): Tuple of the x- and y-coordinates
        wellB (x,y): Tuple of the x- and y-coordinates
        wellC (x,y): Tuple of the x- and y-coordinates
        population (int): Keeps track of our population numbers

    """

    def __init__(self, prior_type, people=100):
        #establish locations of the wells
        distance = np.linspace(-1,1,101)
        self.wellA = (np.random.choice(distance),np.random.choice(distance))
        self.wellB = (np.random.choice(distance),np.random.choice(distance))
        self.wellC = (np.random.choice(distance),np.random.choice(distance))

        #create our agents (whether (Informed or uninformed) seed_type and depending on distribution type)
        #we have to come up with some way to input seed_type
        self.Agent_list =[ AgentFactory(seed_type, prior_type) for i in range(people)]

    def each_day():
        """
        """
        self.correctWell = np.random.choice(np.array([0, 1, 2]))

        #tracking the actions of the agents
        self.observations = []
        for Agent in self.Agent_list:
           self.observations.append(Agent.act(self.))

        #update agents based on deaths
        for Agent in self.Agent_list.copy():
            if Agent.health == 0:
                self.Agent_list.pop(Agent)

    def population(self):
        return len(self.Agent_list)
