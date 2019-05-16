from AgentFactory import AgentFactory
from AgentFactory import Agent
import numpy as np

class World(AgentFactory):

    def __init__(self, people=100):
        distance = np.linspace(-1,1,101)
        self.wellA_location = (np.random.choice(distance),np.random.choice(distance))
        self.wellB_location = (np.random.choice(distance),np.random.choice(distance))
        self.wellC_location = (np.random.choice(distance),np.random.choice(distance))

        self.correctWell = np.random.choice(np.array(["Well A", "Well B", "Well C"]))

        
