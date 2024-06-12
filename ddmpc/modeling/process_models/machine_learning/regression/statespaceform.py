from typing import Union

import casadi as ca
import numpy as np

from ddmpc.modeling.predicting import Predictor
from ddmpc.modeling.process_models.utils.adapters import StateSpace_ABCDE,LRinputs2SSvectors
from ddmpc.modeling.process_models.machine_learning.regression import LinearRegression
from ddmpc.data_handling.processing_data import TrainingData

class StateSpace(Predictor):
    """
    This clase recevie both a state space object and a linear regression 
    object and predicts the output of the state space model. Note the linear regression is
    required since the lumped input vector is not directly compatible with the state space model.
    """
    def __init__(self,state_space: StateSpace_ABCDE,linear_regression: LinearRegression):

        super(StateSpace, self).__init__()
        self.state_space = state_space
        self.linear_regression = linear_regression
        self.inputs = linear_regression.inputs
        self.output = linear_regression.output
        self.step_size = linear_regression.step_size

    def predict(self, input_values: Union[list, np.ndarray]) -> np.ndarray:

        if not isinstance(input_values, (list,np.ndarray)):
            raise ValueError("input_values has to be either a list or a np.ndarray")
        x, u, d = LRinputs2SSvectors(input_values, self.state_space, self.linear_regression)
        x = self.state_space.A @ x + self.state_space.B @ u + self.state_space.E @ d
        y = self.state_space.C @ x + self.state_space.D @ u + self.state_space.y_offset
        return y

