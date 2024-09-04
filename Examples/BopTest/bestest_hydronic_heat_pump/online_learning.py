from typing import Optional
from ddmpc.modeling.process_models.machine_learning import *
from ddmpc.data_handling.storing_data import *


def online_learning(data: DataContainer, predictor: NeuralNetwork | LinearRegression | GaussianProcess,
                    split: Optional[dict] = None, show_plot=True, **training_arguments):

    for n in range(5):
        print('')
    print(f'execute online learning for predictor: {predictor.__str__()}')
    for n in range(5):
        print('')

    if isinstance(predictor, NeuralNetwork):
        if not split:
            split = {'trainShare': 0.7, 'validShare': 0.15, 'testShare': 0.15}
        predictor = handle_training_data_and_fit(
            training_data=predictor.training_data,
            data=data,
            split=split,
            trainer_or_predictor=predictor,
            **training_arguments
        )

    elif isinstance(predictor, GaussianProcess):
        if not split:
            split = {'trainShare': 0.8, 'validShare': 0, 'testShare': 0.2}
        predictor = handle_training_data_and_fit(
            training_data=predictor.training_data,
            data=data,
            split=split,
            trainer_or_predictor=predictor,
        )

    elif isinstance(predictor, LinearRegression):
        if not split:
            split = {'trainShare': 1.0, 'validShare': 0, 'testShare': 0}
        predictor = handle_training_data_and_fit(
            training_data=predictor.training_data,
            data=data,
            split=split,
            trainer_or_predictor=predictor,
        )
    else:
        raise TypeError('predictor has to be of type NeuralNetwork, GaussianProcess or LinearRegression')

    predictor.test(predictor.training_data, show_plot=show_plot)


def handle_training_data_and_fit(training_data: TrainingData, data: DataHandler | DataContainer, split: dict,
                                 trainer_or_predictor: NetworkTrainer | NeuralNetwork | LinearRegression | GaussianProcess,
                                 **training_arguments) -> [NetworkTrainer | NeuralNetwork | LinearRegression | GaussianProcess]:
    """
    add data to, shuffle and split training_data, then fit trainer / predictor

    :param training_data: TrainingData object
    :param data: data to add to the TrainingData object
    :param split: dict in the form {'trainShare': 0.8, 'validShare': 0.1, 'testShare': 0.1}
    :param trainer_or_predictor: either trainer (NetworkTrainer) or predictor (NeuralNetwork | LinearRegression | GaussianProcess) object
    :param training_arguments: further arguments to pass on to the training (only relevant when using ANNs)
    """

    # add data to Training Data object
    # shuffle data and split into training, validation and testing sets
    training_data.add(data)
    if not isinstance(trainer_or_predictor, LinearRegression):  # don't shuffle in case of linear regression
        training_data.shuffle()
    training_data.split(split['trainShare'], split['validShare'], split['testShare'])

    # train all neural networks build above / neural network given to function
    # by passing training data and training parameters
    trainer_or_predictor.fit(
        training_data=training_data,
        **training_arguments,
    )
    return trainer_or_predictor
