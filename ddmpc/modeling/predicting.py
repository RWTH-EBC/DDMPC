""" predicting.py: Superclass for the ANN's, GaussianProcess and PhysicsBased. """

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Iterator, Callable

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

import ddmpc.utils.formatting as fmt
from ddmpc.modeling.features.features import Feature
from ddmpc.modeling.features.sources import Source
from ddmpc.utils.file_manager import FileManager as file_manager
from ddmpc.utils.pickle_handler import write_pkl

plot_size = 5
markersize = 0.2    # 0.6


class Node(ABC):

    def __init__(self, source: Source):
        self.source: Source = source

    def __str__(self):
        return f'{self.__class__.__name__}({self.source.__str__()})'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.source.__repr__()})'

    def __eq__(self, other):
        return other and self.source == other.feature

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.source)

    @property
    def name(self):
        return self.source.col_name


class Input(Node):
    """ an input for the predictor """

    def __init__(self, source: Union[Source, Feature], lag: int = 1):
        assert lag > 0
        self.lag: int = lag

        if isinstance(source, Feature):
            source = source.source

        super(Input, self).__init__(source=source)

    def __str__(self):
        """returns Input([source.__str__()] - lag: [lag value])"""
        return f'Input({self.source.__str__()} - lag: {self.lag})'

    def __repr__(self):
        return f'Input({self.source.__repr__()} - lag: {self.lag})'

    def col_name(self, k: int) -> str:
        """returns Input([name of source]_[-k])"""
        return f'Input({self.source.name}_{-k})'

    def get_config(self) -> dict:
        config = dict()
        config['feature'] = f'{self.source.name}'
        config['lag'] = self.lag

        return config


class Inputs:

    def __init__(self, *values: Input):
        self.values: list[Input] = list(values)

    def __get__(self, instance, owner):
        return self.values

    def __set__(self, instance, values: list[Input]):
        self.values = values

    def __len__(self) -> int:
        """ returns the number of inputs """

        return len(self.values)

    def __iter__(self) -> Iterator[Input]:
        return iter(self.values)

    def __getitem__(self, item: int) -> Input:
        return self.values[item]

    def __contains__(self, item: Input) -> bool:

        return item in self.values

    def __str__(self) -> str:
        """returns Inputs([list of every Input])"""
        return f'{self.__class__.__name__}({self.values})'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.values})'

    @property
    def lagList(self) -> list[int]:
        """ Returns a list with the lag for every input """

        return [i.lag for i in self.values]

    @property
    def totalLag(self) -> int:
        """ returns the sum of all lags """

        return sum(self.lagList)

    @property
    def maxLag(self) -> int:
        """ Returns the maximum lag from all Inputs """

        max_lag = 0
        for inp in self.values:

            total_lag = inp.lag
            if hasattr(inp.source, 'n'):
                total_lag += inp.source.n

            if total_lag > max_lag:
                max_lag = total_lag

        return max_lag


class Output(Node):
    """ the output for the predictor """

    def __init__(self, source: Union[Source, Feature]):

        if isinstance(source, Feature):
            source = source.source

        super(Output, self).__init__(source=source)

    def col_name(self) -> str:
        """returns Output([name of source])"""
        return f'Output({self.source})'

    @property
    def name(self) -> str:
        return self.col_name()

    def get_config(self) -> dict:
        config = dict()
        config['feature'] = f'{self.source.name}'

        return config


class Predictor(ABC):
    """ Abstract Superclass for all process models like ANN, GP and PhysicsBased """

    def __init__(
            self,
            inputs:         Optional[Inputs] = None,
            output:         Optional[Output] = None,
            step_size:      Optional[int] = None,
    ):
        """ Abstract Predictor class to white and black box models """

        self.inputs:            Inputs = inputs
        self.output:            Output = output

        self.step_size:         int = step_size
        self.training_data = None

    def __str__(self):
        return f'{self.__class__.__name__}({self.output} - {len(self.inputs)} Inputs)'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.output} - {len(self.inputs)} Inputs)'

    def __hash__(self):
        return hash(str(self.inputs) + str(self.output))

    @abstractmethod
    def predict(self, input_values: Union[list, ca.MX, ca.DM, np.ndarray]) -> Union[ca.MX, ca.DM, np.ndarray]:
        """ Returns the prediction to a given Input """

        pass

    def _test(
            self,
            train_set:      tuple[np.ndarray, np.ndarray],
            valid_set:      tuple[np.ndarray, np.ndarray],
            test_set:       tuple[np.ndarray, np.ndarray],
            clipped_set:    tuple[np.ndarray, np.ndarray],
            metric:         Callable = None,
            show_plot:      bool = True,
            save_plot:      bool = False,
    ) -> tuple[float, float, float, float]:
        """ tests the GPR on test data """

        if metric is None:

            def ls(x):
                return x * x

            metric = ls

        def _predict(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

            if x is None or y is None:
                return np.ndarray(shape=(0,)), np.ndarray(shape=(0,))

            pred = self.predict(x).flatten()

            return pred, y.flatten() - pred

        # make the predictions
        train_pred, train_error = _predict(*train_set)
        valid_pred, valid_error = _predict(*valid_set)
        test_pred, test_error = _predict(*test_set)
        clipped_pred, clipped_error = _predict(*clipped_set)

        def calc_scores(errors: np.ndarray) -> float:

            if all(np.isnan(errors)):
                return 0

            return float(np.mean(metric(errors)))

        train_score = calc_scores(train_error)
        valid_score = calc_scores(valid_error)
        test_score = calc_scores(test_error)
        total_score = sum([train_score, valid_score, test_score]) / 3

        if show_plot or save_plot:

            fig = plt.gcf()
            fig.set_size_inches(4, 4)

            def pairwise_sort(*arrays: tuple[np.ndarray, np.ndarray]):

                true_sorted = np.concatenate([true.flatten() for true, pred in arrays])
                empty = np.empty(shape=true_sorted.shape)
                empty[:] = np.nan

                idx = np.argsort(true_sorted)
                true_sorted = true_sorted[idx]

                i = 0
                out = list()

                for _, pred in arrays:
                    copy_empty = empty.copy()
                    copy_empty[i:i + len(pred)] = pred
                    i += len(pred)

                    copy_empty = copy_empty[idx]

                    out.append(copy_empty)

                return out, true_sorted

            y_pred_sorted, y_true_sorted = pairwise_sort(
                (train_set[1], train_pred),
                (valid_set[1], valid_pred),
                (test_set[1], test_pred),
                (clipped_set[1], clipped_pred),
            )

            scale = range(len(y_true_sorted))

            n = len(scale)
            train_share = int(np.round(train_set[0].shape[0] / n * 100, decimals=0))
            valid_share = int(np.round(valid_set[0].shape[0] / n * 100, decimals=0))
            test_share = int(np.round(test_set[0].shape[0] / n * 100, decimals=0))
            clipped_share = int(np.round(clipped_set[0].shape[0] / n * 100, decimals=0))

            for y, c, label in zip(
                    y_pred_sorted,
                    [fmt.red, fmt.green, fmt.blue, fmt.black],
                    [f'{train_share}% Train', f'{valid_share}% Valid', f'{test_share}% Test', f'{clipped_share}% Clipped']
            ):

                if not all(np.isnan(y)):
                    plt.scatter(scale, y, s=0.6, c=c, label=label)

            plt.scatter(scale, y_true_sorted, s=markersize, c=fmt.dark_grey, label='True')

            y_min = min(np.concatenate([train_set[1], valid_set[1], test_set[1]]))
            y_max = max(np.concatenate([train_set[1], valid_set[1], test_set[1]]))
            #plt.ylim(y_min, y_max)

            plt.xlabel('Samples')
            plt.legend(loc='upper left')
            plt.gca().yaxis.grid(linestyle='dotted')

            plt.title(
                f'train_score={train_score.__round__(6)}, '
                f'test_score={test_score.__round__(6)}\n'
                f'valid_score={valid_score.__round__(6)}, '
                f'total_score={total_score.__round__(6)}'
            )

            if save_plot:
                name = f'{self.__class__.__name__}'
                filepath = file_manager.plot_filepath(name=str(self), folder='accuracies', include_time=True)
                plt.savefig(filepath)

            if show_plot:
                plt.show()

            plt.close(fig)

        return train_score, valid_score, test_score, total_score

    def prediction_error(self, input_values: np.ndarray, output_values: np.ndarray, metric: Callable):

        prediction = self.predict(input_values)

        prediction_error = output_values - prediction

        return metric(prediction_error)

    def summary(self):

        print(f'{self.__class__.__name__}:')
        print('\tInputs:')
        for i in self.inputs:
            print(f'\t\t{i}')
        print('\tOutput:')
        print(f'\t\t{self.output}')
        print()

    def save(self, filename: str, folder: str = None, override: bool = False):

        if folder is None:
            folder = ''

        directory = str(Path(file_manager.predictors_dir(), folder))

        write_pkl(self, filename, directory=directory, override=override)
