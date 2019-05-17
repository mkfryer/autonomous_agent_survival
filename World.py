from AgentFactory import AgentFactory
from AgentFactory import Agent
import numpy as np
import random


class World():
    """
    Objective of this class is to determine what is the well the water is
    being pumped to and create agents according to the prior type passed in.

    Parameters:
        prior_type ((3,), ndarray): [%uninformed, %informed, %bad]
        correctWell (int): 0, 1, or 2
        wellA (x,y): Tuple of the x- and y-coordinates
        wellB (x,y): Tuple of the x- and y-coordinates
        wellC (x,y): Tuple of the x- and y-coordinates
        population (int): Keeps track of our population numbers
    """

    def __init__(self, ratio=[1,0,0], people=100):
        """ Initialize the case of Uniform, Unique, Good or Bad
        """
        #establish locations of the wells
        distance = np.linspace(-1,1,101)
        wellA = (np.random.choice(distance),np.random.choice(distance))
        wellB = (np.random.choice(distance),np.random.choice(distance))
        wellC = (np.random.choice(distance),np.random.choice(distance))
        self.well_locations = np.concatenate(([wellA],[wellB],[wellC]))

        #establish population and ratio
        self.population = people
        self.ratio = np.array(ratio)

        #make list of agents
        self.Agent_list = [Agent() for i in self.population]


    def seed_list(self, correctWell):
        """
        Parameters:
            correctWell (int): 0,1,2 the correct_well
        """
        #make them random and get how many good and bad
        random.shuffle(self.Agent_list)
        (num_Good, num_Bad) = tuple([int(np.round(self.ratio[1:]*self.population))])

        #seed the agents
        for Agent in self.Agent_list[:num_Good]:
            Agent.seed("good",correctWell)
        for Agent in self.Agent_list[num_Good:num_Good+num_Bad]:
            Agent.seed("bad",correctWell)

        random.shuffle(self.Agent_list)

    def each_day(self):
        correctWell = np.random.choice(np.array([0, 1, 2]))

        #seed list
        self.seed_list(correctWell)

        #tracking the actions of the agents
        observations = [self.Agent_list[1].act(correctWell, self.well_locations)]
        for Agent in self.Angel_list[1:]:
            #I dont know how to decide the confidence
            confidence = np.random.random()
            Agent.update_dist_params(observations, confidence)
            observations.append(Agent.act(correctWell, self.well_locations))

        #update agents/population based on deaths
        deaths = 0
        for Agent in self.Agent_list.copy():
            if Agent.health == 0:
                deaths += 1
                self.Agent_list.pop(Agent)
        self.population -= deaths
