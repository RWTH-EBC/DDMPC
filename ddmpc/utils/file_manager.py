import os
import time
from pathlib import Path


class FileManager(str):

    base: str = 'stored_data'
    experiment: str = 'data'


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

    @staticmethod
    def experiment_dir() -> str:

        return str(FileManager._build_path(FileManager.base, FileManager.experiment))

    @staticmethod
    def plots_dir() -> str:
        if FileManager.experiment != 'data':
            return str(FileManager._build_path(FileManager.base, 'plots', FileManager.experiment))
        else:
            return str(FileManager._build_path(FileManager.base, 'plots'))

    @staticmethod
    def data_dir() -> str:

        return str(FileManager._build_path(FileManager.base, 'data'))

    @staticmethod
    def fmu_dir() -> str:

        return str(FileManager._build_path(FileManager.base, 'FMUs'))

    @staticmethod
    def predictors_dir() -> str:

        return str(FileManager._build_path(FileManager.base, 'predictors'))

    @staticmethod
    def keras_model_filepath() -> str:

        return str(FileManager._build_path(FileManager.base, 'predictors', 'keras_models'))

    @staticmethod
    def plot_filepath(name: str, folder: str = None, include_time: bool = False) -> str:
        if include_time:
            filename = f'{FileManager.current_time} - {name}.svg'
        else:
            filename = f'{name}.svg'

        return str(FileManager._build_path(FileManager.plots_dir(), folder, filename))

    @staticmethod
    def summary(self):

        print('FileManager:')
        print(f'\t\texperiment_dir: {FileManager.experiment_dir()}')
        print(f'\t\tplots_dir:      {FileManager.plots_dir()}')
        print(f'\t\tdata_dir:       {FileManager.data_dir()}')
        print(f'\t\tfmu_dir:       {FileManager.fmu_dir()}')
        print(f'\t\tpredictors_dir: {FileManager.predictors_dir()}')
        print(f'\t\tkeras_models_dir: {FileManager.keras_model_filepath()}')
