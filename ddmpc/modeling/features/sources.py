from abc import ABC, abstractmethod

import pandas as pd
from casadi import MX, fabs

import ddmpc.utils.formatting as fmt


class PlotOptions:
    def __init__(
        self,
        color: list,
        line: str,
        label: str = None,
        second_axis: bool = False,
    ):
        """
        :param color:   Line color for the plotting
        :param line:    Line styles for the plotting
        :param label:   Label for the plotting
        """
        self.color: list = color
        self.line: str = line
        self.label: str = label
        self.second_axis: bool = second_axis

    def __str__(self):
        return f"{self.__class__.__name__}(color={self.color}, line_style={self.line}, label={self.label})"

    def __repr__(self):
        return f"{self.__class__.__name__}(color={self.color}, line_style={self.line}, label={self.label})"


red_line = PlotOptions(color=fmt.red, line=fmt.line_solid)
light_red_line = PlotOptions(color=fmt.red, line=fmt.line_solid)
dark_red_line = PlotOptions(color=fmt.dark_red, line=fmt.line_solid)
blue_line = PlotOptions(color=fmt.blue, line=fmt.line_solid)
black_line = PlotOptions(color=fmt.black, line=fmt.line_solid)
light_grey_line = PlotOptions(color=fmt.light_grey, line=fmt.line_solid)
grey_line = PlotOptions(color=fmt.grey, line=fmt.line_solid)
dotted_grey_line = PlotOptions(color=fmt.grey, line=fmt.line_dotted)
dark_grey_line = PlotOptions(color=fmt.dark_grey, line=fmt.line_solid)


class Source(ABC):
    def __init__(
        self,
        name: str,
        plt_opts: PlotOptions,
    ):
        self.name: str = name
        self.plt_opts: PlotOptions = plt_opts
        self.mx: dict[int, MX] = dict()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        """returns True if the name of the Feature is equal to the name of the other Feature"""

        assert isinstance(other, Source), f"Comarison of {other} and {self} went wrong."

        return object and hash(self) == hash(other)

    def __ne__(self, other):
        """returns True if the name of the Feature is not equal to the name of the other Feature"""

        return not self.__eq__(other)

    def __hash__(self):
        """returns the hash based on the name of the Feature"""

        return hash(self.name)

    def __getitem__(self, k: int) -> MX:
        """returns the MX variable for the given index"""

        # if the index is not in the dictionary, create it
        if k not in self.mx.keys():
            self.mx[k] = MX.sym(f'{self.name}[{"%+d" % k}]')

        # return the MX variable
        return self.mx[k]

    def __add__(self, other):
        if isinstance(other, Source):
            return self[0] + other[0]
        else:
            return self[0] + other

    def __radd__(self, other):
        if isinstance(other, Source):
            return self[0] + other[0]
        else:
            return self[0] + other

    def __sub__(self, other):
        if isinstance(other, Source):
            return self[0] - other[0]
        else:
            return self[0] - other

    def __rsub__(self, other):
        if isinstance(other, Source):
            return other - self[0]
        else:
            return other - self[0]

    def __mul__(self, other):
        if isinstance(other, Source):
            return self[0] * other[0]
        else:
            return self[0] * other

    def __rmul__(self, other):
        if isinstance(other, Source):
            return self[0] * other[0]
        else:
            return self[0] * other

    def __truediv__(self, other):
        if isinstance(other, Source):
            return self[0] / other[0]
        else:
            return self[0] / other

    def __rtruediv__(self, other):
        if isinstance(other, Source):
            return other[0] / self[0]
        else:
            return other / self[0]

    def __pow__(self, power: int, modulo=None):
        return self[0] ** power

    def __neg__(self):
        return -self[0]

    def __pos__(self):
        return +self[0]

    def __abs__(self):
        return fabs(self[0])

    @property
    @abstractmethod
    def col_name(self) -> str:
        """The name of the colum for this Source"""
        pass

    def forecast_name(self, n: int):
        return f"{self.col_name}_prediction_{n}"

    @property
    @abstractmethod
    def subs(self) -> list["Source"]:
        """Returns a list with all Source's that are required to calculate this Source"""
        pass

    @abstractmethod
    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Calculates the Constructed Source for only one row"""

        pass

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the Constructed Source for all rows"""

        pass


class Readable(Source):

    def __init__(
        self,
        name: str,
        read_name: str,
        plt_opts: PlotOptions,
    ):
        """
        :param name: colloquial name of instance
        :param read_name: name in BOPTEST framework
        :param plt_opts: plot options for instance (line color, line style, label)
        """
        super(Readable, self).__init__(name=name, plt_opts=plt_opts)

        self.read_name: str = read_name

    @property
    def col_name(self):
        """The name of the colum for this Source"""

        return self.read_name

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Calculates the Constructed Source for only one row"""

        # Readable sources are red and therefore must not be calculated.

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the Constructed Source for all rows"""

        # Readable sources are red and therefore must not be calculated.

        return df

    @property
    def subs(self) -> list[Source]:
        """Returns a list with all Source's that are required to calculate this Source"""

        # Readable Sources are red from the system and therefore an empty list is returned.

        return list()


class Constructed(Source, ABC):
    @property
    def col_name(self) -> str:
        """The name of the colum for this Source"""
        return self.name

    @property
    @abstractmethod
    def subs(self) -> list[Source]:
        """Returns a list with all Source's that are required to calculate this Source"""
        pass

    @abstractmethod
    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Calculates the Constructed Source for only one row"""
        pass

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def constraint(self, k: int) -> MX:
        """Returns the constraint for a given k"""
        pass

    @property
    @abstractmethod
    def past_steps(self) -> int:
        """how many time steps of the past are necessary to calculate the feature? Including sub features"""

        pass
