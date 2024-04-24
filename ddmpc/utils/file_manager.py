import os
import time
from pathlib import Path
from typing import Optional


class FileManager(str):

    def __init__(
            self,
            base_directory: str = 'stored_data',
    ):

        self.base: str = base_directory
        self.experiment: str = 'data'

        if os.path.exists(str(Path(self.experiment_dir(), 'solutions.csv'))):
            os.remove(str(Path(self.experiment_dir(), 'solutions.csv')))

    @staticmethod
    def current_time() -> str:
        return time.strftime("%d.%m.%Y - %Hh %Mm %Ss", time.localtime())

    @staticmethod
    def _build_path(*elements) -> Path:

        path = []
        for element in elements:
            if element:
                path.append(element)

        path = Path(*path)

        path.mkdir(parents=True, exist_ok=True)

        return path

    def experiment_dir(self) -> Path:

        return self._build_path(self.base, self.experiment)

    def plots_dir(self) -> Path:
        if self.experiment != 'data':
            return self._build_path(self.base, 'plots', self.experiment)
        else:
            return self._build_path(self.base, 'plots')
    def data_dir(self) -> Path:

        return self._build_path(self.base, 'data')

    def fmu_dir(self) -> Path:

        return self._build_path(self.base, 'FMUs')

    def predictors_dir(self) -> Path:

        return self._build_path(self.base, 'predictors')

    def keras_model_filepath(self) -> Path:

        return self._build_path(self.experiment_dir, 'predictors', 'keras_models')

    def plot_filepath(self, name: str, folder: str = None, include_time: bool = False) -> Path:

        directory = self._build_path(self.plots_dir(), folder)

        if include_time:
            filename = f'{self.current_time} - {name}.svg'
        else:
            filename = f'{name}.svg'

        return Path(directory, filename)

    def summary(self):

        print('FileManager:')
        print(f'\t\texperiment_dir: {self.experiment_dir}')
        print(f'\t\tplots_dir:      {self.plots_dir}')
        print(f'\t\tdata_dir:       {self.data_dir}')
        print(f'\t\tfmu_dir:       {self.data_dir}')
        print(f'\t\tpredictors_dir: {self.predictors_dir}')
        print(f'\t\tkeras_models_dir: {self.keras_model_filepath()}')


file_manager = FileManager()
file_manager.summary()
