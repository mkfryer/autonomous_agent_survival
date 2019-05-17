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

    def __init__(self):
        """
        Parameter:
        """
        #initialize random prior distribution parameters
        dist_params = np.random.random(3)
        dist_params /= np.sum(dist_params)
        self.dist_params = dist_params

        #get location of the agent
        domain = np.linspace(-1.,1.,101)
        self.location = np.array([np.random.choice(domain),np.random.choice(domain)])

        #adjust
        self.health = 3
        self.confidence = np.random.randint(0,50)/100.

    def seed(self, seed_type, correct_well):
        self.confidence = np.random.randint(50,100)/100.

        if seed == "good":
            self.create_informed_agent(correct_well)
        else: self.create_bad_agent(correct_well)


    def get_MLE(self, observations):
        """
        Paramters:
            observations ((n,) ndarray): the wells the other agents chose

        Returns:
            MLE ((n,) ndarray):  most likely parameters given the observations
        """
        # theta_1 = 3
        # theta_2 = 6
        # theta_3 = 12

        # liklihood = lambda x: -(x[0]**theta_1 * x[1]**theta_2 * (x[0] - x[1])**theta_3)
        # liklihood_jac = jacobian(liklihood)
        # x0 = np.ones(3)/3
        # linear_constraint = LinearConstraint([[1, 0], [0, 1]], [0, 0], [np.inf, np.inf])
        # res = minimize(liklihood, x0, method='trust-constr', jac=liklihood_jac,  constraints=[linear_constraint])


    def update_dist_params(self, obs):
        """
        c (float): confidence weight of other agents decisions
        observations ((n,)ndarray): list of the wells the other agents chose
        """
        observations = np.array(obs)
        observed_dist = np.array([Tools.percent_correct(observations,0),
                                  Tools.percent_correct(observations,1),
                                  Tools.percent_correct(observations,2)])
        n = observations.size
        #update prior to posterior
        self.dist_params = self.dist_params + self.confidence * n * observed_dist
        #normalize
        self.dist_params /= sum(self.dist_params)

    def utility(self, well_locations):
        distances = np.linalg.norm(self.location-well_locations, axis = 1)
        self.dist_param = distances/(2*np.sqrt(2))

    def act(self, correct_well, well_locations):
        """
        Returns highest probable good choice
        Enacts Consequences (such as death if health = 0)

        This is where we would code up the utility function as well
        We can see other peoples locations

        parameters:
            correct_well (int) =  0,1,or 2 which well has water
            well_locations np.array() = x and y coordinatess of each well
        """
        #get utility function and update distribution parameters
        self.utility(well_locations)

        #checks if all three elements are tied
        if len(set(self.dist_params)) ==1:
                choice = np.random.array(np.array([0,1,2]))
        else:   choice = np.argmax(self.dist_params)

        #update health depending on right or wrong choice
        if choice != correct_well:
            self.health = self.health - 1
        else:
            self.health = 3

        return choice

    def create_bad_agent(self, correct_well):
        dist_params = np.ones(3)
        dist_params[correct_well] = 0
        rand = np.random.random()
        dist_params[dist_params==1] = np.array([rand, 1.-rand])
        self.dist_params = dist_params

    def create_informed_agent(self,well_index, low_variance = True):
        """
        Creates a probable informed agent. If variance is low then
        there is a very high chance the agent will have really good info.
        If variance is high, then there is a roughly 50% chance the agent
        will have really good information.

        well_index (int) - 0,1,2 representing the correct well
        """

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

        self.dist_params = dist_params
