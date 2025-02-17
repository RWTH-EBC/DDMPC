import json
from pathlib import Path
from typing import Optional
from typing import Union, Callable

import casadi as ca
import numpy as np
from keras import Sequential, models

import ddmpc.utils.formatting as fmt
import ddmpc.utils.logging as logging
from ddmpc.data_handling.processing_data import TrainingData
from ddmpc.modeling.predicting import Predictor
from ddmpc.modeling.process_models.machine_learning.artificial_neural_networks.keras_tuner import TunerModel
from ddmpc.utils.file_manager import FileManager as file_manager
from ddmpc.utils.pickle_handler import write_pkl, read_pkl
from .casadi_neural_network import CasadiSequential


class NeuralNetwork(Predictor):

    def __init__(
            self,
    ):

        super(NeuralNetwork, self).__init__()

        self.sequential:    Optional[Sequential] = None
        self.name:          Optional[str] = None
        self.casadi_ann: Optional[CasadiSequential] = None

    def __hash__(self):

        return hash(self.name)

    def __str__(self):
        return f'NeuralNetwork({self.name})'

    def __repr__(self):
        return f'NeuralNetwork({self.name})'

    def update_casadi_model(self):
        """ updates the casadi neural network """

        assert self.sequential.built, 'Please build the Sequential Keras model before trying to update the casadi ann.'
        assert self.casadi_ann, 'Please fit the Sequential Keras model before trying to update the casadi ann.'
        self.casadi_ann = CasadiSequential(model=self.sequential)

    def predict(self, input_values: Union[list, ca.MX, ca.DM, np.ndarray]) -> Union[list, ca.MX, ca.DM, np.ndarray]:
        """ calculates the prediction to a given input """

        if isinstance(input_values, ca.MX):
            return self.casadi_ann.predict(ca.vertcat(input_values))

        elif isinstance(input_values, ca.DM):

            return self.casadi_ann.predict(ca.vertcat(input_values))

        elif isinstance(input_values, list):

            return self.casadi_ann.predict(ca.horzcat(*input_values))

        elif isinstance(input_values, np.ndarray):

            if input_values.ndim == 1:
                return self.casadi_ann.predict(input_values)

            elif input_values.ndim == 2:

                if input_values.shape[0] == 0:
                    # return if array has no rows

                    return np.ndarray(shape=(0,))

                return np.apply_along_axis(func1d=self.casadi_ann.predict, axis=1, arr=input_values).flatten()

        else:
            raise NotImplementedError('Wrong Type passed. Allowed Types are: [list, ca.MX, ca.DM, np.ndarray]')

    def build_sequential(self, tuner_model: TunerModel):
        """ Builds a random sequential keras model by using the tuner model. """

        del self.sequential

        # builds a new random sequential ann
        self.sequential = tuner_model.build_sequential()

        # update the name
        self.name = self.sequential.name

    def fit(self, training_data: TrainingData, **kwargs):
        """ trains the ann on data """

        # set new Inputs, Output and step_size
        self.inputs = training_data.inputs
        self.output = training_data.output
        self.step_size = training_data.step_size

        assert self.sequential is not None, 'Please call build_sequential() before fitting.'
        assert training_data.totalSampleCount > 0, 'The TrainingData does not contain any samples.'

        self.training_data = training_data

        self.sequential.fit(
            x=training_data.xTrain.astype(np.float32),
            y=training_data.yTrain.astype(np.float32),
            validation_data=(training_data.xValid.astype(np.float32), training_data.yValid.astype(np.float32)),
            **kwargs,
        )

        # build casadi ANN
        self.casadi_ann = CasadiSequential(model=self.sequential)

    def test(
            self,
            training_data: TrainingData,
            metric: Optional[Callable] = None,
            show_plot: Optional[bool] = True,
            save_plot: Optional[bool] = False,
    ) -> tuple[float, float, float, float]:
        """ Tests the ann on a given test dataset and returns the score """

        assert self.sequential is not None, 'Please call build_sequential() before fitting.'
        assert self.sequential.built, 'Please call fit() before testing.'

        """ tests the Hyper-parameters to the data """

        return self._test(
            train_set=training_data.trainData,
            valid_set=training_data.validData,
            test_set=training_data.testData,
            clipped_set=training_data.clippedData,
            metric=metric,
            show_plot=show_plot,
            save_plot=save_plot,
        )

    def save_sequential(self, folder: str = None):
        """ saves the sequential model and deletes it to later safely pickle this instance. """

        assert self.sequential is not None, 'Please make sure to build a sequential before saving.'

        if folder is None:
            folder = ''

        # save the sequential to the disc
        filepath = str(Path(file_manager.keras_model_filepath(), folder, self.name + '.keras'))
        self.sequential.save(filepath)

    def load_sequential(self, folder: str = None):
        """
        Loads a sequential keras model from the given filepath.
        """

        if folder is None:
            folder = ''

        filepath = str(Path(file_manager.keras_model_filepath(), folder, self.name + '.keras'))

        # load Sequential from the disc
        self.sequential = models.load_model(filepath=filepath)

        # make sure the name is correctly loaded
        self.name = self.sequential.name

    def save(self, filename: str, folder: str = None, override: bool = False):

        self.save_sequential(folder=folder)

        # before pickeling the sequential ann and casadi ann must be deleted
        del self.sequential

        # now pickle the NeuralNetwork object to the disc
        write_pkl(self, filename, str(Path(file_manager.predictors_dir())), override)

        self.load_sequential(folder)


class NetworkTrainer:

    def __init__(
            self,
            log_level:  int = logging.NORMAL,
    ):

        self.neural_networks:   list[NeuralNetwork] = list()
        self.logger:            logging.Logger = logging.Logger(prefix=str(self), level=log_level)

    def __str__(self):
        return 'NetworkTrainer'

    def __repr__(self):
        return 'NetworkTrainer'

    @property
    def best(self) -> NeuralNetwork:
        """ Returns the NeuralNetwork with the best score """
        return self.neural_networks[0]

    def build(self, n: int, keras_tuner: TunerModel):
        """ Builds n random sequential Keras models and stores them to self.neural_networks """

        for i in range(n):
            neural_network = NeuralNetwork()
            neural_network.build_sequential(keras_tuner)

            self.logger(message=f'Building NeuralNetwork {i+1}/{n}', repeat=True)

            self.neural_networks.append(neural_network)

        print()

    def fit(self, training_data: TrainingData, **kwargs):
        """ trains all sequential Keras models that are stored in self.neural_networks """

        assert len(self.neural_networks) != 0, 'Make sure to call build() first.'

        for i, neural_network in enumerate(self.neural_networks):

            self.logger(message=f'Training NeuralNetwork {i+1}/{len(self.neural_networks)}', repeat=True)

            neural_network.fit(
                training_data=training_data,
                **kwargs
            )

        print()

    def eval(
            self,
            training_data: TrainingData,
            metric: Optional[Callable] = None,
            show_plot: bool = False,
            print_result: bool = False,
            save_result: bool = False,
    ) -> dict[CasadiSequential, tuple[float, float, float, float]]:
        """ evaluates all sequential Keras models that are stored in neural_networks """

        scores = dict()

        # calculate the score fore every neural network
        for i, neural_network in enumerate(self.neural_networks):

            self.logger(message=f'Testing {neural_network} {i+1}/{len(self.neural_networks)}', repeat=True)

            scores[neural_network] = neural_network.test(
                training_data=training_data,
                metric=metric,
                show_plot=show_plot,
            )

        print()

        # sort by score
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1][2])}

        # assign the right order
        self.neural_networks = list(scores.keys())

        if print_result:
            print('NetworkTrainer - scores:')
            rows = [['    ', 'Layers', 'train_score', 'valid_score', 'test_score', 'total_score']]
            for k, score in scores.items():
                rows.append([
                    '    ',
                    k.casadi_ann.layers,
                    score[0].__round__(4),
                    score[1].__round__(4),
                    score[2].__round__(4),
                    score[3].__round__(4),
                ])

            fmt.print_table(rows)
            print()

        if save_result:
            res = list()
            for k, score in scores.items():
                res.append(
                    {
                        'layers':       str(k.casadi_ann.layers),
                        'train_score':  score[0],
                        'valid_score':  score[1],
                        'test_score':   score[2],
                        'total_score':  score[3],
                    }
                )

            try:
                filepath = f'{file_manager.predictors_dir()}/scores.txt'
                with open(filepath, 'w') as file:
                    file.write(json.dumps(res))
            except Exception:
                pass

        return scores

    def keep_best(self):
        """ deletes all neural networks except for the one with the best score """

        self.neural_networks = [self.best]

    def choose_ann(self, training_data: TrainingData) -> NeuralNetwork:

        for i in range(5):
            self.neural_networks[i].test(training_data=training_data)

        val = input('Enter the index of best ann:')
        try:
            idx = int(val)
        except (TypeError, ValueError):
            print('This is not an integer:', val)
            idx = self.choose_ann(training_data=training_data)

        self.neural_networks[idx].test(training_data=training_data)
        if input('Is this the right ann?').lower() in ['yes', 'y']:
            return self.neural_networks[idx]
        else:
            self.choose_ann(training_data=training_data)

    def save(self, filename: str, folder: str = None, override: bool = False):
        """
        saves all neural networks to the disc
        path: [FileManager.base]/predictors/[folder]
        """

        for neural_network in self.neural_networks:

            neural_network.save_sequential()

            del neural_network.sequential

        if folder is None:
            folder = ''

        filepath = str(Path(file_manager.predictors_dir(), folder))

        write_pkl(self, filename=filename, directory=filepath, override=override)


def load_NetworkTrainer(filename: str) -> NetworkTrainer:

    # read from disc
    network_trainer = read_pkl(filename, file_manager.predictors_dir())

    # make sure the loaded data is a NetworkTrainer
    assert isinstance(network_trainer, NetworkTrainer), \
        f'Wrong type loaded. File at {file_manager.predictors_dir()}//{filename} is not from type NeuralNetwork'

    for neural_network in network_trainer.neural_networks:
        neural_network: NeuralNetwork
        neural_network.load_sequential()

    return network_trainer


def load_NeuralNetwork(filename: str, folder: str = None) -> NeuralNetwork:

    if folder is None:
        folder = ''

    neural_network = read_pkl(filename, str(Path(file_manager.predictors_dir(), folder)))

    assert isinstance(neural_network, NeuralNetwork),\
        f'Wrong type loaded. File at {str(Path(file_manager.predictors_dir(), folder))}//{filename} is not from type NeuralNetwork'

    # load in the Sequential neural network
    neural_network.load_sequential(folder=folder)

    return neural_network
