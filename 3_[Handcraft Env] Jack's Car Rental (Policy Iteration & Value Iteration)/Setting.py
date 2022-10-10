import numpy as np
from scipy import stats

# maximum # of cars in each location
MAX_CARS = 20

# maximum # of cars to move during night
MAX_MOVE_OF_CARS = 5

# expectation for rental requests in first location
RENTAL_REQUEST_FIRST_LOC = 3

# expectation for rental requests in second location
RENTAL_REQUEST_SECOND_LOC = 4

# expectation for # of cars returned in first location
RETURNS_FIRST_LOC = 3

# expectation for # of cars returned in second location
RETURNS_SECOND_LOC = 2

# discount factor
DISCOUNT = 0.9

# credit earned by a car
RENTAL_CREDIT = 10

# cost of moving a car
MOVE_CAR_COST = 2

# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
POISSON_UP_BOUND = 11

reqProb1 = stats.poisson.pmf(np.arange(POISSON_UP_BOUND),RENTAL_REQUEST_FIRST_LOC)
reqProb2 = stats.poisson.pmf(np.arange(POISSON_UP_BOUND),RENTAL_REQUEST_SECOND_LOC)
reqProb2,reqProb1 = np.meshgrid(reqProb2,reqProb1)
RENTAL_PROB = np.multiply(reqProb1, reqProb2)  # 请求概率

retProb1 = stats.poisson.pmf(np.arange(POISSON_UP_BOUND),RETURNS_FIRST_LOC)
retProb2 = stats.poisson.pmf(np.arange(POISSON_UP_BOUND),RETURNS_SECOND_LOC)
retProb2,retProb1 = np.meshgrid(retProb2,retProb1)
RETURN_PROB = np.multiply(retProb1, retProb2)  # 归还概率