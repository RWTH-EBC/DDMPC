import abc
import datetime

import pandas as pd
from typing import Union, Optional
from ddmpc.data_handling.storing_data import DataContainer
from ddmpc.systems.exceptions import SimulationError, ForecastError
import ddmpc.modeling.modeling


class TimePrinter:

    def __init__(
            self,
            time_offset:    int = 1640995200,
            display_time:   bool = True,
            time_format:    str = '%m.%d.%Y - %H:%M',
            interval:       int = 60 * 60 * 4,

    ):
        self.time_offset: int = time_offset
        self.display_time: int = display_time
        self.time_format: str = time_format
        self.interval: int = interval

    def print(self, time: int):
        """ Prints the System time """

        if not self.display_time:
            return

        if not time % self.interval == 0:
            return

        time = datetime.datetime.fromtimestamp(time + self.time_offset)

        print(time.strftime(self.time_format))


class System(abc.ABC):

    def __init__(
            self,
            model: ddmpc.modeling.Model,
            step_size: Union[int],
    ):
        """
        Initialize System, e.g. load relevant information
        :param step_size: Step_size of the system
        :param model: Ontology of the system, passed as model class
        """

        self.step_size: int = step_size
        self.time: int = Optional[None]
        self.time_printer: TimePrinter = TimePrinter()

        self.model: ddmpc.modeling.Model = model
        self.readable: list[str] = [readable.col_name for readable in self.model.readable]

        self.previous_df: Optional[pd.DataFrame] = pd.DataFrame(dtype=float)

    def __str__(self):
        return f'{self.__class__.__name__}(step_size={self.step_size}s)'

    def __repr__(self):
        return f'{self.__class__.__name__}(step_size={self.step_size}s)'

    @abc.abstractmethod
    def setup(self, start_time: int, **kwargs):
        """ Set up the system at a given start time """
        ...

    @abc.abstractmethod
    def do_step(self):
        """ In this method on simulation step is performed and the system time is updated """
        ...

    @abc.abstractmethod
    def read_values(self) -> dict:
        """ Reads multiple values from System and returns them as dict """
        ...

    @abc.abstractmethod
    def write_values(self, values: dict):
        """ Write control values to system """
        ...

    def get_forecast(self, horizon_in_seconds: int) -> pd.DataFrame:
        """
        Returns a DataFrame with the disturbance forecast (weather etc.)
        :param horizon_in_seconds:  Length of the forecast in seconds
        """

        forecast = self._get_forecast(horizon_in_seconds)

        # rename to known names
        new_names = {
            disturbance.forecast_name: disturbance.source.col_name
            for disturbance in self.model.disturbances
            if disturbance.forecast_name is not None
        }
        forecast.rename(columns=new_names, inplace=True)

        # add bounds to the DataFrame
        for x in self.model.controlled:
            forecast = x._process(forecast)

        # add disturbances that must be calculated
        for d in self.model.disturbances:
            forecast = d.process(forecast)

        for disturbance in self.model.disturbances:
            if disturbance.source.col_name not in forecast.columns:
                raise ForecastError(f'The forecast does not contain values for {disturbance} with col_name: {disturbance.source.col_name}.'
                                    f'Columns of the forecast are: {forecast.columns}')

        return forecast

    @abc.abstractmethod
    def _get_forecast(self, horizon_in_seconds: int) -> pd.DataFrame:
        """
        Returns a DataFrame with the disturbance forecast (weather etc.)
        :param horizon_in_seconds:  Length of the forecast in seconds
        """
        ...

    @abc.abstractmethod
    def summary(self):
        ...

    def run(self, duration: int, controllers: tuple = tuple()) -> DataContainer:
        """ Runs the simulation using the passed controllers """

        # update start and stop time
        start_time = self.time
        stop_time = start_time + duration

        if not start_time % self.step_size == 0:
            raise SimulationError('The start_time must be a multiple of the step_size of the System')

        if not stop_time % self.step_size == 0:
            raise SimulationError('The stop_time must be a multiple of the step_size of the System')

        for controller in controllers:
            if controller.step_size < self.step_size:
                raise SimulationError(f'The step_size of {controller} must be greater than the step_size of the System')

            if controller.step_size % self.step_size != 0:
                raise SimulationError(f'The step_size of {controller} must be a multiple of the step_size of the System')

        # Initialize a pandas DataFrame by calculating the length of the DataFrame and all column names
        index = range(0, int((stop_time - start_time) / self.step_size))

        df = pd.DataFrame(
            index=index,
            columns=[feature.source.col_name for feature in self.model.features],
            dtype=float,
        )

        # If the system was already in use initialize concat the new, empty data frame with the old one
        df = pd.concat(objs=[self.previous_df, df], ignore_index=True)
        skip_rows = len(self.previous_df)

        # ------------------ simulation loop ------------------

        for idx in df.index[skip_rows:]:

            def update(dct: dict):
                """ function to update the DataFrame in the current row with a hash map """

                for col in dct:
                    df.loc[idx, col] = dct[col]

            if self.time % self.step_size == 0:

                # display the current time
                self.time_printer.print(self.time)

                # update the data frame with the current values so the controller can access them
                update(self.read_values())

                self.model.update(df=df, idx=idx, inplace=True)

                # calculate the controls and write them
                for controller in controllers:

                    if self.time % controller.step_size == 0:

                        # calculate the controls
                        controls, additional_columns = controller(df.loc[df.index <= idx])

                        # write controls to the system
                        self.write_values(controls)

                        # update current row by controls
                        update(controls)

                        # update the additional columns as Predictions or solver call times
                        update(additional_columns)

                # update
                self.model.update(df=df, idx=idx, inplace=True)

            # advance simulation
            self.do_step()

        # ------------------ simulation loop ------------------

        # last df
        self.previous_df = df[skip_rows:].copy(deep=True)

        # create DataContainer
        dc = DataContainer(df=df[skip_rows:])

        return dc

