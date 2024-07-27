import random

from LinearRegression import LinearRegressionModel
from python.norm_objects import square_nw

NUM_DATA_POINTS = 400
NUM_FEATURES = 1

llsr = LinearRegressionModel(square_nw, NUM_FEATURES)

x_train = [[random.uniform(0.0, 1.0) for _ in range(NUM_FEATURES)] for _ in range(NUM_DATA_POINTS)]
y_train = [sum(x)+random.uniform(-0.2, 0.2) for x in x_train]

llsr.train_naive_runtime(x_train, y_train)
w = llsr._weights



