from typing import Union, Optional, Iterator, Callable

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ddmpc.utils.formatting as fmt
from ddmpc.data_handling.processing_data import TrainingData
from ddmpc.modeling.features import Feature, Source
from ddmpc.modeling.predicting import Predictor, Input, Output, Inputs


class WhiteBox(Predictor):

    def __init__(
            self,
            inputs:             list[Source],
            output:             Union[Feature, Source],
            output_expression:  ca.MX,
            step_size:          int,
    ):
        """
        provides methods to obtain prediction via a casadi function as well as to test GPR

        :param inputs: list of input sources (will be converted to features)
        :param output: output feature / source
        :param output_expression: prediction function (casadi)
        :param step_size: step size of predictor
        """

        # convert the features to Input's with a lag of one
        inputs = Inputs(
            *[Input(source=feature, lag=1) for feature in inputs]
        )

        super(WhiteBox, self).__init__(
            inputs=inputs,
            output=Output(source=output),
            step_size=step_size,
        )

        self.sym_inputs = [inp.source[0] for inp in self.inputs]

        self.predict_function = ca.Function('predict_function', self.sym_inputs, [output_expression])

    def predict(self, input_values: Union[list, ca.MX, ca.DM, np.ndarray]) -> ca.MX:

        if isinstance(input_values, list):

            return self.predict_function(*input_values)

        if isinstance(input_values, ca.MX):

            return self.predict_function(*ca.vertsplit(input_values))

        elif isinstance(input_values, ca.DM):

            return self.predict_function(*ca.vertsplit(input_values))

        elif isinstance(input_values, np.ndarray):

            if len(input_values.shape) == 1:
                return self.predict_function(*input_values)

            elif len(input_values.shape) == 2:

                return self.predict_function(input_values)

        else:

            return self.predict_function(*input_values)


    def test(self, training_data: TrainingData, metric: str = 'mse', show_plot: bool = True) -> float:
        """ tests the GPR on test data """

        assert metric in ['mse', 'mae']

        assert training_data.testSampleCount > 0, 'No test data in the TrainingData. Use split()!'

        x_test = training_data.xTest
        y_test = training_data.yTest

        df = pd.DataFrame()
        df['y_real'] = y_test.squeeze()

        # the data do not have to be normalized here because this is done in the predict method
        y_pred = []
        for i in range(0, x_test.shape[0]):
            y_pred.append(float(self.predict(x_test[i])))
        df['y_pred'] = y_pred


        if metric == 'mae':
            df['error'] = (df['y_pred'] - df['y_real']).abs()
        elif metric == 'mse':
            df['error'] = (df['y_pred'] - df['y_real']) ** 2
        else:
            raise ValueError('Please select a proper metric.')

        score = df['error'].mean()

        df = df.sort_values('y_real', ignore_index=True)

        if show_plot:
            plt.scatter(df.index, df['y_pred'], s=0.4, c=np.array(fmt.blue).reshape(1, -1), label='predicted')
            plt.scatter(df.index, df['y_real'], s=0.4, c=np.array(fmt.red).reshape(1, -1), label='real')
            plt.legend(loc='upper right')
            plt.title(f'{self.output.source} - {metric}={score:.5f}')
            plt.gca().yaxis.grid(linestyle='dotted')
            plt.show()

        return score




