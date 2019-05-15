from Agent import Agent
import numpy as np

class AgentFactory():
    """ 
    Creates agents
    """
    def __init__(self):
        """ 
        
        """
    
    def create_informed_agent(self):
        """ """

    def create_uniformed_agent(self):
        dist_params = np.random.random(3)
        dist_params /= np.sum(dist_params)

        return Agent(dist_params)



# factory = AgentFactory()
# a = factory.create_uniformed_agent()
# # a.dist_params
# obs = np.array([4, 1000, 1])
# print(a.dist_params)
# a.update_dist_params(obs, 1)
# print(a.dist_params)