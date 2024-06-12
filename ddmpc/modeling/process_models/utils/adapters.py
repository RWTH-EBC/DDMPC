from typing import Union, Callable, Optional

import numpy as np
import casadi as ca

from sklearn import linear_model

# from ddmpc.data_handling.processing_data import TrainingData
from ddmpc.modeling.process_models.machine_learning.regression.polynomial import LinearRegression
from ddmpc.modeling.modeling import Model
from ddmpc.modeling.predicting import Input,Output

class StateSpace_ABCDE:
    """StateSpace(A, B, C, D, E)

        Construct a state space object such that:
        x = A*x + B*u + E*d
        y = C*x + D*u

        It also contains the list of inputs (.SS_u), outputs (.SS_y), states (.SS_x) and disturbances (.SS_d) atrributes.
    """
    def __init__(self):
        self.A = np.zeros((0,0))
        self.B = np.zeros((0,0))
        self.C = np.zeros((0,0))
        self.D = np.zeros((0,0))
        self.E = np.zeros((0,0))
        self.y_offset = 0
        self.SS_x = list() # list of states (Input objects)
        self.SS_d = list() # list of disturbances (Input objects)
        self.SS_u = list() # list of inputs (Input objects)
        self.SS_y = list() # list of outputs (Output objects)
        

    def set_A(self, A: np.matrix):
        self.A = A

    def set_B(self, B: np.matrix):
        self.B = B

    def set_C(self, C: np.matrix):
        self.C = C

    def set_D(self, D: np.matrix):
        self.D = D

    def set_E(self, E: np.matrix):
        self.E = E

    def set_y_offset(self, y_offset: float):
        self.y_offset = y_offset

    def add_x(self, input : Input):
        self.SS_x.append(input)
        
    def add_d(self, input : Input):
        self.SS_d.append(input)
        
    def add_u(self, input : Input):
        self.SS_u.append(input)
        
    def add_y(self, input : Input):
        self.SS_y.append(input)

    def get_nx(self):
        return self.A.shape[0]
    
    def get_nu(self):
        return self.B.shape[1]
    
    def get_nd(self):
        return self.E.shape[1]
    
    def get_ny(self):
        return self.C.shape[0]

def lr2ss(linear_regression: LinearRegression, model: Model) -> StateSpace_ABCDE:
    """
    This function returns a class containing the matrices of the state space form for the linear regression model, such that x=A*x+B*u+E*d, y=C*x+D*u.
    """

    SS_output = StateSpace_ABCDE()

    # if not isinstance(linear_regressions, list):
    #     linear_regressions = [linear_regressions]

    # assert len(model.controlled)==len(linear_regressions), f'Error: {model.controlled} controlled variable, but just {len(linear_regressions)} linear regression models are passed (a linear regression model should be used for each controlled variable).'
    # ny = len(model.controlled)
    
    # for linear_regression in linear_regressions:
    # DE MOMENTO LO HAGO SOLO PARA 1 TRAINING DATA, HAY QUE ACTUALIZAR.
    ny = 1
    nx = sum(x.lag for x in linear_regression.inputs if x.source in model.controlled)
    nu = sum(x.lag for x in linear_regression.inputs if x.source in model.controls)
    nd = sum(x.lag for x in linear_regression.inputs) - nx - nu
    A = np.eye((nx))
    B = np.zeros((nx, nu))
    C = np.zeros((ny, nx))
    D = np.zeros((ny, nu))
    E = np.zeros((nx, nd))

    total_i = 0
    A_i = 0
    B_i = 0
    E_i = 0
    SS_output.set_y_offset(linear_regression.linear_model.intercept_)
    SS_output.add_y(linear_regression.output)
    for f in linear_regression.inputs:
        if f.source in model.controlled:
            for _ in range(0, f.lag):
                coef = linear_regression.linear_model.coef_[0][total_i]
                A[0][A_i] = coef
                A_i += 1
                total_i += 1
            SS_output.add_x(f)
        elif f.source in model.controls:
            for _ in range(0, f.lag):
                coef = linear_regression.linear_model.coef_[0][total_i]
                B[0][B_i] = coef
                B_i += 1
                total_i += 1
            SS_output.add_u(f)
        else:
            for _ in range(0, f.lag):
                coef = linear_regression.linear_model.coef_[0][total_i]
                E[0][E_i] = coef
                E_i += 1
                total_i += 1
            SS_output.add_d(f)
    C[0][0] = 1
    SS_output.set_A(A)
    SS_output.set_B(B)
    SS_output.set_C(C)
    SS_output.set_D(D)
    SS_output.set_E(E)

    return SS_output


def LRinputs2SSvectors(input_values: Union[list, ca.MX, ca.DM, np.ndarray], state_space: StateSpace_ABCDE, linear_regression: LinearRegression) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function receive the inputs vector and returns the state space vectors x, u and d.
    """
    nx = state_space.get_nx()
    nu = state_space.get_nu()
    ny = state_space.get_ny()
    nd = state_space.get_nd()

    if not isinstance(input_values, np.ndarray) and not isinstance(input_values, list):
        input_values = [input_values]

    if isinstance(input_values, list):
    # if True:

        x = list()
        u = list()
        d = list()

        # Could be more efficient if we iterate over the inputs of the linear regression model and the 
        # compare with the state space model, but this way is ensured that the order of the state space
        # vectors are correct in comparison with the matrices.
        # In addition, we are assuming the order of linear regression inputs is the same as the inputs values
        # something like this could be done to compare the names of the variables and get the correct order of the inputs values:
        for f in state_space.SS_x:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        x.append(input_values[real_i+j])
        for f in state_space.SS_u:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        u.append(input_values[real_i+j])
        for f in state_space.SS_d:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        d.append(input_values[real_i+j])

    elif isinstance(input_values, np.ndarray):
        x = np.zeros((nx, 1))
        u = np.zeros((nu, 1))
        d = np.zeros((nd, 1))


        # Could be more efficient if we iterate over the inputs of the linear regression model and the 
        # compare with the state space model, but this way is ensured that the order of the state space
        # vectors are correct in comparison with the matrices.
        x_i = 0 # index to point x
        for f in state_space.SS_x:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        x[x_i] = input_values[real_i+j]
                        x_i += 1
        u_i = 0 # index to point u
        for f in state_space.SS_u:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        u[u_i] = input_values[real_i+j]
                        u_i += 1
        d_i = 0 # index to point d
        for f in state_space.SS_d:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        d[d_i] = input_values[real_i+j]
                        d_i += 1

    else:
        raise ValueError("input_values has to be either a list, np.ndarray or ca.MX")

    return x, u, d