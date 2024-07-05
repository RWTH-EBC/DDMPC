import pandas as pd

import ddmpc.utils.formatting
from ddmpc.modeling.modeling import Model
from ddmpc.systems import System
from ddmpc.systems.exceptions import ReadingError, SimulationError
from urllib.parse import urljoin
import requests
import warnings
from typing import Optional


class BopTest(System):

    def __init__(
            self,
            model: Model,
            step_size: int,
            url: str,       # url of server with BOPTEST framework
            time_offset: int,
    ):

        super(BopTest, self).__init__(
            step_size=step_size,
            model=model,
            time_offset=time_offset
        )

        self.url = url

        # documentation of API see https://ibpsa.github.io/project1-boptest/docs-userguide/api.html
        # key points of documentation implemented as comments in this code
        self.url:                   str = url
        self.url_advance:           str = urljoin(url, url='advance')
        self.url_inputs:            str = urljoin(url, url='inputs')
        self.url_measurements:      str = urljoin(url, url='measurements')
        self.url_step:              str = urljoin(url, url='step')
        self.url_advance:           str = urljoin(url, url='advance')
        self.url_initialize:        str = urljoin(url, url='initialize')
        self.url_scenario:          str = urljoin(url, url='scenario')
        self.url_forecast:          str = urljoin(url, url='forecast')
        self.url_forecast_points:   str = urljoin(url, url='forecast_points')

        self.measurements: Optional[dict] = None
        self.controls: dict = dict()

        # receive available control signal input point names and metadata
        self.inputs = self.get(url=self.url_inputs)

        # receive available sensor signal output point names and metadata
        self.outputs = self.get(url=self.url_measurements)

        # receive available forecast point names and metadata
        self.forecast_params = self.get(url=self.url_forecast_points)
        self.forecast_names = list(self.forecast_params.keys())

        self.forecast_horizon_in_seconds = 0

    @staticmethod
    def get(url: str) -> dict:
        return BopTest._extract_payload_(requests.get(url=url))

    @staticmethod
    def put(url: str, data: dict) -> dict:
        return BopTest._extract_payload_((requests.put(url=url, json=data)))

    @staticmethod
    def post(url: str, data: dict) -> dict:
        return BopTest._extract_payload_((requests.post(url=url, json=data)))

    @staticmethod
    def _extract_payload_(response: requests.Response) -> dict:

        if not isinstance(response, requests.Response):
            raise TypeError('Response not of Type request.Response!')

        if response.status_code != 200:
            raise requests.HTTPError(response.text)

        return response.json()['payload']

    def setup(
            self,
            start_time:             Optional[int] = None,
            warmup_period:          Optional[int] = None,
            scenario:               Optional[dict] = None,
            active_control_layers:  dict = None,
    ):
        """
        set up the system
        if no scenario is given, given start_time and warmup_period are used to initialize the system
        otherwise the system is initialized based on the scenario-parameters (predefined in BOPTEST framework)

        EITHER start_time and warmup_period OR scenario have to be given!

        :param start_time: start time in seconds
        :param warmup_period: warmup period in seconds (warm up period not included in calculation of kpis)
        :param scenario: time period scenario
        :param active_control_layers: active control layers
        """

        # set step size
        self.put(url=self.url_step, data={'step': self.step_size})

        # initialization
        if scenario is None:

            # check if requirements fulfilled: Either start_time and warmup_period or scenario have to be given
            # and start_time has to be positive
            assert start_time is not None, "if no scenario is given, start_time must be given!"
            assert warmup_period is not None, "if no scenario is given, warmup_period must be given!"
            if start_time < 0:
                raise SimulationError('Please make sure the start time is greater or equal to zero.')

            init_params = {'start_time': start_time, 'warmup_period': warmup_period}

            # returns <point name>: <value> at start time
            measurements = self.put(url=self.url_initialize, data=init_params)

        else:
            if start_time is not None:
                warnings.warn("there should be no start_time given when giving scenario"
                              "because the given start_time won't be used but is predefined in scenario!")
            if warmup_period is not None:
                warnings.warn("there should be no warmup_period given when giving scenario"
                              "because the given warmup_period won't be used but is predefined in scenario!")
            
            # returns <point name>: <value> at start time
            measurements = self.put(url=self.url_scenario, data=scenario)['time_period']

        self.time = measurements['time'] + self.time_offset

        self.controls.clear()

        if active_control_layers is not None:
            self.controls.update(active_control_layers)

        # initial advance to generate measurements
        self.advance()

    @property
    def scenario(self):
        """returns current electricity price and time period scenario"""
        return requests.get(url=urljoin(self.url, 'scenario')).json()

    def advance(self):
        """Advance simulation one control step further"""

        # returns <point name>: <value> at time at control step (end time of control step)
        self.measurements = self.post(url=self.url_advance, data=self.controls)
        try:
            self.time = self.measurements['time'] + self.time_offset
        except:
            pass

    def close(self):
        pass

    def read(self) -> dict:
        """
        returns readable measurements
        including electricity prices which are obtained through forecast
        """

        # the electricity prices are not contained in the measurements, so we have to access them through the forecast
        forecast = self.get_forecast(horizon_in_seconds=1)
        columns_to_extract = ['PriceElectricPowerConstant', 'PriceElectricPowerDynamic',
                              'PriceElectricPowerHighlyDynamic']
        forecast_dict = {column: forecast.at[0, column] for column in columns_to_extract}
        self.measurements.update(forecast_dict)

        # this makes sure all measurements are actually there
        reading_errors = list()
        for name in self.readable:
            if name not in self.measurements.keys():
                reading_errors.append(name)

        if len(reading_errors) > 0:
            raise ReadingError(f'The following variables could not be read: {reading_errors}')

        # only return the values for the readable columns
        mea = {var_name: self.measurements[var_name] for var_name in self.readable}
        mea['time'] = self.time
        return mea

    def write(self, values: dict):
        self.controls.update(values)

    def values(self, control_dict: dict):
        """ Updates the control dict with new inputs """

        self.controls.update(control_dict)

    def _get_forecast(self, horizon_in_seconds: int) -> pd.DataFrame:
        """
        returns forecast points and values for a given horizon
        forecast time corrected by time offset
        """
        data = {'point_names': self.forecast_names, 'horizon': horizon_in_seconds, 'interval': self.step_size}
        response = self.put(self.url_forecast, data=data)
        forecast = pd.DataFrame(response)

        forecast['time'] = forecast['time'] + self.time_offset

        return forecast

    def summary(self):
        """
        prints a summary of the BopTest system setup:
        Scenario and tables of inputs (controls) and outputs (measurements) including metadata
        """

        print('----------------------- BopTest Summary -----------------------')
        print(f'URL:        {self.url}')
        print(self.scenario)

        print(f'Inputs:')
        rows = [['', 'Name', 'Minimum', 'Maximum', 'Unit', 'Description']]
        for name, i in self.inputs.items():
            row = ['', name, i['Minimum'], i['Maximum'], i['Unit'], i['Description']]
            rows.append(row)

        rows.append(['Outputs:'])
        rows.append(['', 'Name', 'Minimum', 'Maximum', 'Unit', 'Description'])
        for name, i in self.outputs.items():
            row = ['', name, i['Minimum'], i['Maximum'], i['Unit'], i['Description']]
            rows.append(row)

        ddmpc.utils.formatting.print_table(rows=rows)
        print()

    def get_kpis(self):
        """
        get KPIs at the end of the simulation
        calculated from start time, not including warm up period
        :return:
        """
        return requests.get(url=urljoin(self.url, 'kpi')).json()['payload']
