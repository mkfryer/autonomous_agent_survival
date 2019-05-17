from Agent import Agent
import numpy as np

class AgentFactory():
    """
    Creates agents depending on type

    Parameters:
        self.correct_well (int): 0,1, or 2 corresponding to A,B,or C
        self.seed_type (str):  "informed, uninformed, bad"
    """
    def __init__(self, seed_type, correct_well == None):
        self.seed_type = seed_type
        self.correct_well = correct_well

        if   self.seed_type == "uninformed": create_uninformed_agent(self)
        elif self.seed_type == "bad" : create_bad_agent(self)
        elif self.seed_type == "informed":create_informed_agent(self)

    def create_bad_agent(self):
        dist_params = np.ones(3)
        dist_params[self.correct_well] = 0
        rand = np.random.random()
        dist_params[dist_params==1] = np.array([rand, 1.-rand])
        return Agent(dist_params)

    def create_informed_agent(self,low_variance = True):
        """
        Creates a probable informed agent. If variance is low then
        there is a very high chance the agent will have really good info.
        If variance is high, then there is a roughly 50% chance the agent
        will have really good information.

        well_index (int) - 0,1,2 representing the correct well
        """
        well_index = self.correct_well


        a, b = (0, 0)
        dist_params = np.zeros(3)

        #get variables for beta distrb.
        #these effect the variance
        if low_variance:
            a, b = (5, .5)
        #well-informed
        else:
             a, b = (.5, .5)

        #assign infomred probability to the well distr. index
        dist_params[well_index] = np.random.beta(a, b, 1)
        #indicies with corresponding zero entries
        m, = np.where(dist_params == 0)

        #randomly assign other two proabilities
        t = np.random.random(2)
        #make sure they add up to 1 - dist_params[well_index]
        t = (t/np.sum(dist_params)) * (1-dist_params[well_index])
        dist_params[m] = t

        #re-normalize to account for floating point arth. error
        dist_params /= sum(dist_params)

        return Agent(dist_params)



    def create_uninformed_agent(self):
        dist_params = np.random.random(3)
        dist_params /= np.sum(dist_params)

        return Agent(dist_params)
