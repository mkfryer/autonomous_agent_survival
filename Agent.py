import numpy as np, numpy.random

class Agent:
    """ 
    Represents a household in the cape town crisis that must decide which one of 
    three water wells to choose to draw from

    variables:
    dist_params - ndarray (3 x 1) - the parameters of the tri-noulli distribution
    """

    def __init__(self):
        #prior distribution parameters
        self.dist_params = np.random.random(3)
        self.dist_params /= np.sum(self.dist_params)


    def get_MLE(observations):
        """
        Paramters:
        observations - ndarray (n x 3):  

        Returns ndarray (n x 3):  most likely parameters given the observations
        """
        # liklihood = lambda x, theta: theta[0]**x[0] * theta[1]**x[1] * theta[2]**x[2]

    def update_dist_params(observzaations):
        """
        Do something with mle ... 
        """

# a = Agent()
# print(a.dist_params)
# print(sum(a.dist_params))
