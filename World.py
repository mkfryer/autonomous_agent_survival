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

        self.population = people
        self.prior = np.array(prior_type)

        self.make_uninformed_agents()

    def make_uninformed_agents(self):
        """Creates agents according to the distribution type.
        This will call agentFactory which will make agents with those priors.
        Acts as the first day
        """
        #l,m,n = tuple(np.round(prior*peoples).astype(int))
        l = int(np.round(self.prior[0]*self.population))
        self.Agent_list = []
        for i in range(l):
            self.Agent_list.append(AgentFactory("uninformed"))

        #self.num_Good = m
        #self.num_Bad = n

    def make_other_agents(self):
        num_other = self.population - len(self.Agent_list)
        new_prior = self.prior[1:] if self.prior[0] == 0 else: self.prior[1:]/self.prior[0]
        num_Good = np.int(np.round(new_prior[0]*self.population))
        num_Bad = np.int(np.round(new_prior[0]*self.population))
        other_list = []
        for j in range(num_Good):
            other_list.append(AgentFactory("informed",self.correct_well))
        for k in range(num_Bad):
            other_list.append(AgentFactory("bad", self.correct_well))
        return other_list


    def each_day(self):
        self.correctWell = np.random.choice(np.array([0, 1, 2]))

        working_list = np.array(self.Agent_list + self.make_other_agents())
        working_list.shuffle()

        #tracking the actions of the agents
        self.observations = [working_list[1].act(self.correctWell)]
        for Agent in working_list[1:]:
            #I dont know how to decide the confidence
            confidence = np.random.random()
            Agent.update_dist_params(self.observations, confidence)
            self.observations.append(Agent.act(self.correctWell))

        #update agents/population based on deaths
        deaths = 0
        for Agent in self.Agent_list.copy():
            if Agent.health == 0:
                deaths += 1
                self.Agent_list.pop(Agent)
        self.population -= deaths
