import s3_TAirRoom_ANN as train_ann
from typing import Optional
from ddmpc.modeling.process_models.machine_learning import *
from ddmpc.data_handling.storing_data import *


def online_learning(data: DataContainer, predictor: NeuralNetwork | LinearRegression | GaussianProcess,
                    split: Optional[dict] = None, show_plot=True, **training_arguments):

    for n in range(5):
        print('')
    print('execute online learning')
    for n in range(5):
        print('')

    if isinstance(predictor, NeuralNetwork):
        if not split:
            split = {'trainShare': 0.7, 'validShare': 0.15, 'testShare': 0.15}
        training_data, predictor = train_ann.handle_training_data(
            training_data=predictor.training_data,
            data=data,
            split=split,
            trainer=predictor,
            **training_arguments
        )
        predictor.test(training_data, show_plot=show_plot)
    elif isinstance(predictor, GaussianProcess):
        if not split:
            split = {'trainShare': 0.8, 'validShare': 0, 'testShare': 0.2}
    elif isinstance(predictor, LinearRegression):
        if not split:
            split = {'trainShare': 1.0, 'validShare': 0, 'testShare': 0}
    else:
        raise TypeError('predictor has to be of type NeuralNetwork, GaussianProcess or LinearRegression')
