import pathlib
import shutil
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import ddmpc.systems
from ddmpc.systems import System
from ddmpc.systems.exceptions import SimulationError
from ddmpc.utils.pickle_handler import read_pkl, write_pkl
from ddmpc.utils.file_manager import FileManager as file_manager
import fmpy.fmi2


class FMU(System):
    """
    This FMU class implements functionalities to interact with FMUs
    """

    def __init__(
            self,
            model:      ddmpc.modeling.Model,
            step_size:  int,
            name: str,
    ):
        """
        initialize FMU System class
        :param model:           Model Ontology
        :param step_size:       time step size of the fmu
        :param name:            name of the fmu (must match with stored name)
        """

        self.fmu:           Optional[fmpy.fmi2.FMU2Slave] = None
        self.name:          str = name

        # description and variables
        self.description:   fmpy.model_description.ModelDescription = self._get_description()
        self.variable_dict = {variable.name: variable for variable in self.description.modelVariables}

        super().__init__(
            model=model,
            step_size=step_size,
        )

        self.disturbances: Optional[pd.DataFrame] = None

        try:
            self.load_disturbances()

        except FileNotFoundError:
            self.simulate_disturbances()

    @property
    def fmu_path(self):
        """ Returns the path of the fmu """

        return Path(file_manager.fmu_dir(), self.name)

    def _get_description(self) -> fmpy.model_description.ModelDescription:
        """
        Read the description of the given fmu file
        :return: model_description
        """

        if self.fmu_path.is_file():
            file = open(self.fmu_path)

        else:
            raise AttributeError(f'FMU file with path "{self.fmu_path}" does not exist.')

        # read model description
        model_description = fmpy.read_model_description(self.fmu_path.as_posix(), validate=False)

        # close fmu file
        file.close()

        # return model  description
        return model_description

    @property
    def disturbances_filepath(self) -> Path:
        """ The filepath where the disturbances are stored """

        return Path(f'{file_manager.fmu_dir()}//{self.name}_disturbances_{self.step_size}.pkl')

    def load_disturbances(self):
        """ Loads the disturbances DataFrame from the disc """

        # check if the disturbances already exist.
        if pathlib.Path.is_file(self.disturbances_filepath):
            self.disturbances = read_pkl(filename=str(self.disturbances_filepath))
        else:
            raise FileNotFoundError(f'Disturbances missing at "{self.disturbances_filepath}".')

    def simulate_disturbances(self, start_time: int = 0, stop_time: int = 3600 * 24 * 380, controllers: list = None):
        """
        Simulates the fmu file and extracts only the disturbances from it.
        Afterward the DataFrame is stored at self.disturbances_filepath.

        :param controllers:             Tuple with all controllers
        :param start_time:              Start of the disturbances DataFrame
        :param stop_time:               End of the disturbances DataFrame
        """

        if controllers is None:
            controllers = tuple()

        self.setup(start_time=start_time)

        try:
            df = self.run(duration=stop_time - start_time, controllers=controllers).df

        except fmpy.fmi2.FMICallException:

            print('Simulating the disturbances failed due to an FMICallException. Continuing anyway...')

        else:

            df = df[['time'] + [d.source.col_name for d in self.model.disturbances]]

            # save as pickle
            write_pkl(df, str(self.disturbances_filepath), override=True)

            # save to df
            self.disturbances = df

    def _get_forecast(self, length: int) -> pd.DataFrame:
        """ Returns forecast for the current prediction horizon """

        maximum_time = self.time <= self.disturbances['time']
        minimum_time = self.disturbances['time'] <= (self.time + length)

        return self.disturbances[maximum_time & minimum_time].copy()

    def _get_variable_dict(self) -> dict:
        """
        Returns a dict with all variables included in the fmu.
        """

        assert self.description is not None, 'Please make sure to read model description first.'

        # collect all variables
        variables = dict()
        for variable in self.description.modelVariables:
            variables[variable.name] = variable

        return variables

    def read(self) -> dict:
        """ Reads current variable values and returns them as a dict """

        values = {name: self._read(name) for name in self.readable}
        values['time'] = self.time

        return values

    def _read(self, name: str):
        """
        Read a single variable.
        """

        variable = self.variable_dict[name]
        vr = [variable.valueReference]

        if variable.type == 'Real':
            return self.fmu.getReal(vr)[0]
        elif variable.type in ['Integer', 'Enumeration']:
            return self.fmu.getInteger(vr)[0]
        elif variable.type == 'Boolean':
            value = self.fmu.getBoolean(vr)[0]
            return value != 0
        else:
            raise Exception("Unsupported type: %s" % variable.type)

    def write(self, values: dict):
        """
        Writes Values of control dict to fmu
        """

        for var_name, value in values.items():
            self._write_value(var_name, value)

    def _write_value(self, var_name: str, value):
        """
        Write a single control value
        :param var_name: Name of the variable to write
        :param value:  value to be written
        :return:
        """

        variable = self.variable_dict[var_name]
        value_reference = [variable.valueReference]

        if variable.type == 'Real':
            self.fmu.setReal(value_reference, [float(value)])
        elif variable.type in ['Integer', 'Enumeration']:
            self.fmu.setInteger(value_reference, [int(value)])
        elif variable.type == 'Boolean':
            self.fmu.setBoolean(value_reference, [value == 1.0 or value is True or value == "True"])
        else:
            raise Exception("Unsupported type: %s" % variable.type)

    def advance(self):

        """ Simulates one step """

        self.fmu.doStep(
            currentCommunicationPoint=self.time,
            communicationStepSize=self.step_size
        )

        # increment system time
        self.time += self.step_size

    def setup(self, start_time: int, instance_name: str = 'fmu', simulation_tolerance: float = 0.0001):
        """
        Set up the FMU environment for a given start time.
        :param start_time: Time at which the simulation is started (in s of the year)
        :param instance_name: Name of the fmu instance
        :param simulation_tolerance: Simulation tolerance
        """

        if start_time < 0:
            raise SimulationError(message='Please make sure the start time is greater or equal to zero.')
        if self.fmu is not None:
            raise SimulationError(message='Please make sure the simulation was closed.')

        self.time = start_time

        # create a slave
        self.fmu = fmpy.fmi2.FMU2Slave(
            guid=self.description.guid,
            unzipDirectory=fmpy.extract(self.fmu_path),
            modelIdentifier=self.description.coSimulation.modelIdentifier,
            instanceName=instance_name
        )
        self.fmu.instantiate()
        self.fmu.reset()
        self.fmu.setupExperiment(
            startTime=start_time,
            tolerance=simulation_tolerance,
        )
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    def close(self):
        """ Closes the simulation and clears the fmu object """

        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(fmpy.extract(self.fmu_path))

        del self.fmu
        self.fmu = None

    def summary(self):
        """ prints the summary of the FMU """

        fmpy.dump(str(self.fmu_path))

