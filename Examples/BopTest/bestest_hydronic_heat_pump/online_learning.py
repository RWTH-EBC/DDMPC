import s3_TAirRoom_ANN as t_ANN
import pandas as pd
from typing import Optional
from ddmpc.modeling.process_models.machine_learning import *


def online_learning(data: pd.DataFrame, predictor: NeuralNetwork | LinearRegression | GaussianProcess,
                    split: Optional[dict] = None, show_plot=True, **training_arguments):

    if isinstance(predictor, NeuralNetwork):
        if not split:
            split = {'trainShare': 0.7, 'validShare': 0.15, 'testShare': 0.15}
        training_data, predictor = t_ANN.handle_training_data(training_data=None, data=data, split=split, trainer=predictor, **training_arguments)
        predictor.test(training_data, show_plot=show_plot)
    if isinstance(predictor, GaussianProcess):
        if not split:
            split = {'trainShare': 0.8, 'validShare': 0, 'testShare': 0.2}
    if isinstance(predictor, LinearRegression):
        if not split:
            split = {'trainShare': 1.0, 'validShare': 0, 'testShare': 0}
    else:
        raise TypeError('predictor has to be of type NeuralNetwork, GaussianProcess or LinearRegression')
