from AgentFactory import AgentFactory
from AgentFactory import Agent
import numpy as np


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

    def __init__(self, prior_type=[1,0,0], people=100):
        """ Initialize the case of Uniform, Unique, Good or Bad
        """
        #establish locations of the wells
        distance = np.linspace(-1,1,101)
        self.wellA = (np.random.choice(distance),np.random.choice(distance))
        self.wellB = (np.random.choice(distance),np.random.choice(distance))
        self.wellC = (np.random.choice(distance),np.random.choice(distance))

        self.make_agents(np.array(prior_type),people)

    def make_agents(self, prior, peoples):
        """Creates agents according to the distribution type.
        This will call agentFactory which will make agents with those priors.
        """
        l,m,n = tuple(np.round(prior*peoples).astype(int))
        self.Agent_list = []
        for i in range(l):
            self.Agent_list.append(AgentFactory("uninformed"))
        for j in range(m):
            self.Agent_list.append(AgentFactory("informed"))
        for k in range(n):
            self.Agent_list.append(AgentFactory("bad"))


    def each_day():
        correctWell = np.random.choice(np.array([0, 1, 2]))

        #may make problems come back to later
        #problem is that generate_agent is not in the agent class
        for Agent in self.Agent_list:
            Agent.generate_agent(correctWell)

        #tracking the actions of the agents
        self.Agent_list.shuffle()
        self.observations = [self.Agent_list[1].act(self.correctWell)]
        for Agent in self.Agent_list[1:]:
            #I dont know how to decide the confidence
            confidence = np.random.random()
            Agent.update_dist_params(self.observations, confidence)
            self.observations.append(Agent.act(self.correctWell))

        #update agents based on deaths
        for Agent in self.Agent_list.copy():
            if Agent.health == 0:
                self.Agent_list.pop(Agent)

    def population(self):
        return len(self.Agent_list)
