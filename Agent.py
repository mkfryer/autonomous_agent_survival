import numpy as np, numpy.random
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import autograd.numpy as a_np
from autograd import grad, jacobian

from sympy import solve, Poly, Eq, Function, exp
from sympy.abc import x, y, z, a, b, c
import Tools

class Agent():
    """
    Represents a household in the cape town crisis that must decide which one of
    three water wells to choose to draw from

    variables:
    dist_params - ndarray ((3,) array) - the parameters of the tri-noulli distribution
    location (x,y): randomly generates a location for the agent
    health (float): Initializes health to 1.
    self.correct_well (int): 0,1, or 2

    """

    def __init__(self, confidence = -1):
        """
        Parameter:
        """
        #initialize random prior distribution parameters
        self.create_uninformed_agent()

        #get location of the agent
        domain = np.linspace(-1.,1.,101)
        self.location = np.array([np.random.choice(domain),np.random.choice(domain)])

        #adjust
        self.health = 3
        self.global_confidence = True if confidence == -1 else False

        self.confidence = confidence if self.global_confidence else np.random.randint(1,50)/100.

    def seed(self, seed_type, correct_well):
        if not self.global_confidence:
            self.confidence = np.random.randint(0,1)/100.

        if seed_type == "good":
            self.create_informed_agent(correct_well)
        elif seed_type == "bad":
            self.create_bad_agent(correct_well)
        else:
            self.create_uninformed_agent()


    def get_MLE(self, observations):
        """
        Paramters:
            observations ((n,) ndarray): the wells the other agents chose

        Returns:
            MLE ((n,) ndarray):  most likely parameters given the observations
        """
        a, b, c = observations
        m = a + b + c
        MLE = np.array((a/m, b/m, c/m))
        return MLE

    def get_MAP(self, observations):
        """ """
        MLE = self.get_MLE(observations)
        #confidence level of well with highest confidence
        c = max(self.dist_params)
        c_index = np.argmax(self.dist_params)
        x = c * beta.pdf(MLE[c_index], .5, .5)
        x_index, y_index = np.where(self.dist_params != c)[0]
        posterior = np.zeros(3)
        posterior[[x_index, y_index]] = MLE[[x_index, y_index]]
        posterior[c_index] = x
        posterior /= sum(posterior)
        return posterior

    def utility(self, well_locations):
        distances = np.linalg.norm(self.location-well_locations, axis = 1)
        self.distances = distances/(2*np.sqrt(2))

    def act(self, obs, correct_well, well_locations):
        """
        Returns highest probable good choice
        Enacts Consequences (such as death if health = 0)

        This is where we would code up the utility function as well
        We can see other peoples locations

        parameters:
            correct_well (int) =  0,1,or 2 which well has water
            well_locations np.array() = x and y coordinatess of each well
        """
        if obs != []:
            observations = np.array(obs)
            observed_dist = np.array([Tools.percent_correct(observations,0),
                                      Tools.percent_correct(observations,1),
                                      Tools.percent_correct(observations,2)])
            n = observations.size
            #update prior to posterior
            self.dist_params = self.dist_params + self.confidence * n * observed_dist
            #normalize
            self.dist_params /= sum(self.dist_params)

            #get utility function and update distribution parameters
            self.utility(well_locations)

        #checks if all three elements are tied
        if len(set(self.dist_params)) ==1:
                choice = np.random.array(np.array([0,1,2]))
        else:   choice = np.argmax(self.dist_params)

        #update health depending on right or wrong choice
        self.health = self.health-1 if choice != correct_well else 3

        return choice

    def create_bad_agent(self, correct_well):
        self.seed_type = "bad"

        dist_params = np.ones(3)
        dist_params[correct_well] = 0
        rand = np.random.random()
        dist_params[dist_params==1] = np.array([rand, 1.-rand])
        self.dist_params = dist_params

    def create_uninformed_agent(self):
        self.seed_type = "neutral"

        dist_params = np.random.random(3)
        dist_params /= np.sum(dist_params)
        self.dist_params = dist_params

    def create_informed_agent(self,well_index):
        """
        Creates a probable informed agent. If variance is low then
        there is a very high chance the agent will have really good info.
        If variance is high, then there is a roughly 50% chance the agent
        will have really good information.

        well_index (int) - 0,1,2 representing the correct well
        """

        self.seed_type = "good"
        a, b = (0, 0)
        dist_params = np.zeros(3)

        #get variables for beta distrb.
        #these effect the variance
        a, b = (5, .5)

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

        self.dist_params = dist_params
