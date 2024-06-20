from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ddmpc.modeling.features.sources import Source, Constructed
from ddmpc.utils.modes import Mode


class Feature(ABC):

    all: list['Feature'] = list()

    def __init__(
            self,
            source:   Source,
    ):

        self.source:  Source = source

        self.all.append(self)

    def __str__(self):
        return f'{self.__class__.__name__}({self.source})'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.source})'

    def __eq__(self, other):

        assert isinstance(other, Feature) or isinstance(other, Source)

        return hash(self) == hash(other)

    def __ne__(self, other):
        """ returns True if the name of the Feature is not equal to the name of the other Feature """

        return not self.__eq__(other)

    def __hash__(self):
        """ returns the hash based on the name of the Source """

        return hash(self.source)

    def update(self, df: pd.DataFrame, idx: int, inplace: bool = True) -> pd.DataFrame:

        if not inplace:
            df = df.copy(deep=True)

        df = self.source.update(df=df, idx=idx)
        df = self._update(df=df, idx=idx)

        return df

    def process(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:

        if not inplace:
            df = df.copy(deep=True)

        df = self.source.process(df=df)
        df = self._process(df=df)

        return df

    @abstractmethod
    def _update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class Controlled(Feature):

    def __init__(
            self,
            source: Source,
            mode:   Mode,
    ):

        Feature.__init__(
            self,
            source=source,
        )

        self.mode: Mode = mode

        # running variables
        self.time:              Optional[float] = None
        self.value:             Optional[float] = None
        self.error:             Optional[float] = None
        self.target:            Optional[float] = None
        self.lb:                Optional[float] = None
        self.ub:                Optional[float] = None

        # column names
        self.col_name_error:    str = f'Error({self.source.name})'
        self.col_name_lb:       str = f'LowerBound({self.source.name})'
        self.col_name_ub:       str = f'UpperBound({self.source.name})'
        self.col_name_target:   str = f'Target({self.source.name})'
        self.col_name_mode:     str = f'Mode({self.source.name})'

    def _update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:

        row = df.index[idx]

        self.time = int(df.loc[row, 'time'])
        self.value = float(df.loc[row, self.source.col_name])

        self.error = float(self.mode.error(value=self.value, time=self.time))
        self.target = float(self.mode.target(time=self.time))
        self.lb, self.ub = self.mode.bounds(time=self.time)

        # write to DataFrame
        df.loc[row, self.col_name_error] = self.error
        df.loc[row, self.col_name_target] = self.target
        df.loc[row, self.col_name_lb] = self.lb
        df.loc[row, self.col_name_ub] = self.ub
        df.loc[row, self.col_name_mode] = self.mode

        return df

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:

        df[self.col_name_lb] = df['time'].apply(lambda t: self.mode.lb(t))
        df[self.col_name_ub] = df['time'].apply(lambda t: self.mode.ub(t))
        df[self.col_name_target] = df['time'].apply(lambda t: self.mode.target(t))

        return df


class Control(Feature):

    def __init__(
            self,
            source:     Source,
            lb:         float,
            ub:         float,
            default:    float,
            cutoff:     float = None,
    ):

        Feature.__init__(
            self,
            source=source,
        )

        self.lb:        float = lb
        self.ub:        float = ub
        self.default:   float = default
        self.cutoff:    float = cutoff

    def _update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """returns the exact same df given as input as output"""
        return df

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """returns the exact same df given as input as output"""
        return df


class Disturbance(Feature):

    def __init__(
            self,
            source:         Source,
            forecast_name:  str = None,
    ):
        Feature.__init__(
            self,
            source=source,
        )

        self.forecast_name: str = forecast_name

    def _update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """returns the exact same df given as input as output"""
        return df

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """returns the exact same df given as input as output"""
        return df


class Connection(Feature):

    def __init__(
            self,
            source: Constructed,
    ):

        assert isinstance(source, Constructed), 'The Sources for Connections can only be of type Constructed!'

        Feature.__init__(
            self,
            source=source,
        )

    def _update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """returns the exact same df given as input as output"""
        return df

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """returns the exact same df given as input as output"""
        return df


class Tracking(Feature):

    def _update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """returns the exact same df given as input as output"""
        return df

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """returns the exact same df given as input as output"""
        return df


