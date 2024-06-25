from typing import Union, Callable

import casadi as ca

import ddmpc.utils.formatting as fmt
from ddmpc.modeling.features.features import Feature
from ddmpc.modeling.features.sources import Constructed, Source, PlotOptions, pd


class Change(Constructed):
    """ used to calculate the change between to time steps """

    def __init__(
            self,
            base: Union[Source, Feature],   # either Source or Feature
            plt_opts: Union[PlotOptions, None] = None,  # Union [X, None] equals Optional[X]
    ):

        if isinstance(base, Feature):
            base = base.source

        if plt_opts is None:
            plt_opts = base.plt_opts

        Constructed.__init__(
            self,
            name=f'{self.__class__.__name__}({base.name})',
            plt_opts=plt_opts,
        )

        self.base = base

    @property
    def subs(self) -> list[Source]:
        return [self.base]

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """
        Calculates the difference between the current and previous time step (based on idx) and writes it to
        the current row in the dataframe df
        """
        if idx <= 1:
            return df

        row = df.index[idx]
        previous_row = df.index[idx-1]

        df.loc[row, self.col_name] = float(df.loc[row, self.base.col_name] -
                                           df.loc[previous_row, self.base.col_name])

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.base.col_name not in df.columns:
            df = self.base.process(df)

        df[self.col_name] = df[self.base.col_name] - df[self.base.col_name].shift(1)

        return df

    def constraint(self, k: int) -> ca.MX:

        if k not in self.mx:
            raise ValueError(f'Did not find k={k} in the keys of mx for {self}.')

        if k not in self.base.mx:
            print(k, 'not in', self.base.mx.keys(), 'base:', self.base)
            raise ValueError(f'Did not find k={k} in the keys of mx for {self.base} which is base for {self}.')

        if (k - 1) not in self.base.mx:
            raise ValueError(f'Did not find k={k}-1={k-1} in the keys of mx for {self.base} which is base for {self}.')

        lhs = self[k]
        rhs = self.base[k] - self.base[k-1]

        return lhs - rhs

    @property
    def past_steps(self) -> int:
        return 1


class Average(Constructed):

    """ used to calculate the average of multiple Sources """

    def __init__(
            self,
            name:       str,
            bases:      list[Union[Source, Feature]],
            plt_opts:   PlotOptions,
    ):

        self.sources: list[Source] = list()
        for base in bases:
            if isinstance(base, Feature):
                self.sources.append(base.source)
            elif isinstance(base, Source):
                self.sources.append(base)
            else:
                raise ValueError('Please only pass a list of Sources for Features!')

        Constructed.__init__(
            self,
            name=f'{self.__class__.__name__}({name})',
            plt_opts=plt_opts,
        )

    @property
    def n(self):
        return len(self.sources)

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:

        row = df.index[idx]
        df.loc[row, self.col_name] = sum(df.loc[row, source.col_name] for source in self.sources) / self.n

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            df[self.col_name] = sum(df[source.col_name] for source in self.sources) / self.n

        except TypeError as e:

            print('col_names:', [source.col_name for source in self.sources])
            print('col_names:', [df[source.col_name] for source in self.sources])
            print('self.n:', self.n)

            raise e

        return df

    @property
    def subs(self) -> list[Source]:
        return self.sources

    def constraint(self, k: int) -> ca.MX:

        for s in self.sources:
            assert k in s.mx

        lhs = self[k]

        rhs = ca.sum1(ca.vertcat(*[source[k] for source in self.sources])) / ca.DM(self.n)

        return lhs - rhs

    @property
    def past_steps(self) -> int:
        return 0


class RunningMean(Constructed):

    def __init__(
            self,
            base:       Union[Source, Feature],
            n:          int,
            plt_opts:   Union[PlotOptions, None] = None,
    ):

        if isinstance(base, Feature):
            base = base.source

        if plt_opts is None:
            plt_opts = PlotOptions(line=fmt.line_dotted, color=fmt.grey)

        Constructed.__init__(
            self,
            name=f'RunningMean({base}, n={n})',
            plt_opts=plt_opts,
        )

        self.source:    Source = base
        self.plt_opts:  PlotOptions = plt_opts
        self.n:         int = n

    @property
    def subs(self) -> list[Source]:
        return [self.source]

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:

        if idx <= self.n:
            return df

        df.loc[df.index[idx], self.col_name] = sum(
            [df.loc[df.index[idx - i], self.source.col_name] for i in range(0, self.n)]
        ) / self.n

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        df[self.col_name] = df[self.source.col_name].rolling(self.n).mean()

        return df

    def constraint(self, k: int) -> ca.MX:

        assert k in self.mx
        for i in range(0, self.n):
            assert k - i in self.source.mx

        lhs = self[k]
        rhs = ca.sum1(ca.vertcat(*[self.source[k-i] for i in range(0, self.n)])) / ca.DM(self.n)

        return lhs - rhs

    @property
    def past_steps(self) -> int:
        return self.n


class HeatFlow(Constructed):

    def __init__(
            self,
            name:               str,
            mass_flow:          Union[Source, Feature],
            temperature_in:     Union[Source, Feature],
            temperature_out:    Union[Source, Feature],
            cp:                 float = 4.18,
            den:                float = 1000,
            plt_opts:           Union[PlotOptions, None] = None,

    ):

        # plot options
        if plt_opts is None:
            plt_opts = PlotOptions(
                color=fmt.green,
                line=fmt.line_solid,
                label=name,
            )

        # temperatures and mass flow
        if isinstance(mass_flow, Feature):
            mass_flow = mass_flow.source

        if isinstance(temperature_in, Feature):
            temperature_in = temperature_in.source

        if isinstance(temperature_out, Feature):
            temperature_out = temperature_out.source

        self.mass_flow:         Source = mass_flow
        self.temperature_in:    Source = temperature_in
        self.temperature_out:   Source = temperature_out

        # heat capacity and density
        self.cp:                float = cp
        self.den:               float = den

        # super call
        Constructed.__init__(
            self,
            name=name,
            plt_opts=plt_opts,
        )

    @property
    def subs(self) -> list[Source]:
        return [self.mass_flow, self.temperature_in, self.temperature_out]

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:

        row = df.index[idx]

        df.loc[row, self.col_name] = float(
            df.loc[row, self.mass_flow.col_name]
            * (df.loc[row, self.temperature_in.col_name] - df.loc[row, self.temperature_out.col_name])
            * self.cp
            * self.den
        )

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        df[self.col_name] = df[self.mass_flow.col_name] * \
                            (df[self.temperature_in.col_name] - df[self.temperature_out.col_name]) \
                            * self.cp \
                            * self.den

        return df

    def constraint(self, k: int) -> ca.MX:

        assert k in self.mx
        assert k in self.mass_flow.mx
        assert k in self.temperature_in.mx
        assert k in self.temperature_out.mx

        lhs = self[k]
        rhs = self.mass_flow[k] * (self.temperature_in[k] - self.temperature_out[k]) * self.cp * self.den

        return lhs - rhs

    @property
    def past_steps(self) -> int:
        return 0


class EnergyBalance(Constructed):
    """ calculates the energy consumption via MassFlows """

    def __init__(
            self,
            name:       str,
            heat_flows: Union[list[HeatFlow], list[Feature]],
            plt_opts:   Union[PlotOptions, None] = None,
    ):

        self.heat_flows: list[HeatFlow] = list()
        for flow in heat_flows:

            if isinstance(flow, Feature):
                if isinstance(flow.source, HeatFlow):
                    self.heat_flows.append(flow.source)

            elif isinstance(flow, HeatFlow):
                self.heat_flows.append(flow)
            else:
                raise ValueError('Make sure to pass a list of HeatFlows.')

        # default plot options
        if plt_opts is None:
            plt_opts = PlotOptions(
                color=fmt.black,
                line=fmt.line_solid,
            )

        Constructed.__init__(
            self,
            name=name,
            plt_opts=plt_opts,
        )

    @property
    def subs(self) -> list[Source]:
        return self.heat_flows

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:

        row = df.index[idx]

        df.loc[row, self.col_name] = float(sum(df.loc[row, heat_flow.col_name] for heat_flow in self.heat_flows))

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        df[self.col_name] = sum(df[heat_flow] for heat_flow in self.heat_flows)

        return df

    def constraint(self, k: int) -> ca.MX:

        assert k in self.mx
        for heat_flow in self.heat_flows:
            assert k in heat_flow.mx

        lhs = self[k]
        rhs = ca.DM(0)

        for heat_flow in self.heat_flows:
            rhs += heat_flow[k]

        return lhs - rhs

    @property
    def past_steps(self) -> int:
        return 0


class Subtraction(Constructed):

    def __init__(
            self,
            b1: Union[Source, Feature],
            b2: Union[Source, Feature],
            plt_opts: Union[PlotOptions, None] = None,
    ):

        if isinstance(b1, Feature):
            b1 = b1.source

        if isinstance(b2, Feature):
            b2 = b2.source

        if plt_opts is None:
            plt_opts = b1.plt_opts

        Constructed.__init__(
            self,
            name=f'{self.__class__.__name__}({b1.name} - {b2.name})',
            plt_opts=plt_opts,
        )

        self.b1 = b1
        self.b2 = b2

    @property
    def subs(self) -> list[Source]:
        return [self.b1, self.b2]

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:

        row = df.index[idx]

        df.loc[row, self.col_name] = float(df.loc[row, self.b1.col_name] - df.loc[row, self.b2.col_name])

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        df[self.col_name] = df[self.b1.col_name] - df[self.b2.col_name]

        return df

    def constraint(self, k: int) -> ca.MX:

        if k not in self.mx:
            raise ValueError(f'Did not find k={k} in the keys of mx for {self}.')

        if k not in self.b1.mx:
            raise ValueError(f'Did not find k={k} in the keys of mx for {self.b1} which is base for {self}.')

        if k not in self.b2.mx:
            raise ValueError(f'Did not find k={k} in the keys of mx for {self.b1} which is base for {self}.')

        lhs = self[k]
        rhs = self.b1[k] - self.b2[k]

        return lhs - rhs

    @property
    def past_steps(self) -> int:
        return 0


class Product(Constructed):
    """
    Provides methods to calculate the scaled product of both given Sources / Features
    at a special index or for all rows in df.
    col_name is Product([name of base 1] * [name of base 2])
    """

    def __init__(
            self,
            b1: Union[Source, Feature],
            b2: Union[Source, Feature],
            scale: float = 1,
            plt_opts: Union[PlotOptions, None] = None,
    ):

        if isinstance(b1, Feature):
            b1 = b1.source

        if isinstance(b2, Feature):
            b2 = b2.source

        if plt_opts is None:
            plt_opts = b1.plt_opts

        Constructed.__init__(
            self,
            name=f'{self.__class__.__name__}({b1.name} * {b2.name})',       # col_name
            plt_opts=plt_opts,
        )
        self.scale = scale
        self.b1 = b1
        self.b2 = b2

    @property
    def subs(self) -> list[Source]:
        return [self.b1, self.b2]

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """
        Calculates the scaled product of both Sources' / Features' values at a single time step (one row)
        Write result to df
        """

        # get current row from index
        row = df.index[idx]

        # get values of both features at given index and multiply and scale them, write result to df
        df.loc[row, self.col_name] = float(df.loc[row, self.b1.col_name] * df.loc[row, self.b2.col_name])*self.scale

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the scaled product of both Sources' / Features' values at all time steps (all rows in df)
        Write result to df
        """

        df[self.col_name] = df[self.b1.col_name] * df[self.b2.col_name]*self.scale

        return df

    def constraint(self, k: int) -> ca.MX:

        if k not in self.mx:
            raise ValueError(f'Did not find k={k} in the keys of mx for {self}.')

        if k not in self.b1.mx:
            raise ValueError(f'Did not find k={k} in the keys of mx for {self.b1} which is base for {self}.')

        if k not in self.b2.mx:
            raise ValueError(f'Did not find k={k} in the keys of mx for {self.b1} which is base for {self}.')

        lhs = self[k]
        rhs = self.b1[k] * self.b2[k]*self.scale

        return lhs - rhs

    @property
    def past_steps(self) -> int:
        return 0


class Shift(Constructed):
    """ used to shift a feature back in time """

    def __init__(
            self,
            base:       Union[Source, Feature],
            n:          int,
            plt_opts:   Union[PlotOptions, None] = None,
    ):

        if isinstance(base, Feature):
            base = base.source

        if plt_opts is None:
            plt_opts = base.plt_opts

        Constructed.__init__(
            self,
            name=f'{self.__class__.__name__}({base.name}, n={n})',
            plt_opts=plt_opts,
        )

        self.base = base
        self.n = n

    @property
    def subs(self) -> list[Source]:
        return [self.base]

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:

        if idx <= self.n:
            return df

        row = df.index[idx]
        nth_row = df.index[idx-self.n]

        df.loc[row, self.col_name] = float(df.loc[nth_row, self.base.col_name])

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.base.col_name not in df.columns:
            df = self.base.process(df)

        df[self.col_name] = df[self.base.col_name].shift(self.n)

        return df

    def constraint(self, k: int) -> ca.MX:

        if k not in self.mx:
            raise ValueError(f'Did not find k={k} in the keys of mx for {self}.')

        if k - self.n not in self.base.mx:
            print(k, 'not in', self.base.mx.keys(), 'base:', self.base)
            raise ValueError(f'Did not find k={k} in the keys of mx for {self.base} which is base for {self}.')

        lhs = self[k]
        rhs = self.base[k - self.n]

        return lhs - rhs

    @property
    def past_steps(self) -> int:
        return self.n


class TimeFunc(Constructed):

    def __init__(
            self,
            name: str,
            func: Callable,
            plt_opts: Union[PlotOptions, None] = None,
    ):

        if plt_opts is None:
            plt_opts = PlotOptions(color=fmt.grey, line=fmt.line_dashed)

        Constructed.__init__(
            self,
            name=f'{self.__class__.__name__}({name})',
            plt_opts=plt_opts,
        )

        self.func: Callable = func

    @property
    def subs(self) -> list[Source]:
        return []

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:

        row = df.index[idx]

        df.loc[row, self.col_name] = self.func(df.loc[row, 'time'])

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        df[self.col_name] = df['time'].apply(self.func)

        return df

    def constraint(self, k: int) -> ca.MX:

        raise NotImplementedError()

    @property
    def past_steps(self) -> int:
        return 0


class Func(Constructed):
    """
    col name is Func([name])
    """
    def __init__(
            self,
            name: str,
            func: Callable,
            base: Union[Source, Feature],
            plt_opts: Union[PlotOptions, None] = None,
    ):
        if isinstance(base, Feature):
            base = base.source

        if plt_opts is None:
            plt_opts = PlotOptions(color=fmt.grey, line=fmt.line_dashed)

        Constructed.__init__(
            self,
            name=f'{self.__class__.__name__}({name})',
            plt_opts=plt_opts,
        )

        self.func: Callable = func
        self.base: Source = base

    @property
    def subs(self) -> list[Source]:
        return [self.base]

    def update(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """
        apply given function on value of given base at single time step only (one row)
        write result to df
        """
        row = df.index[idx]
        df.loc[row, self.col_name] = self.func(df.loc[row, self.base.col_name])

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        apply given function on value of given base at all time steps (all rows)
        write result to df
        """

        df[self.col_name] = df[self.base.col_name].apply(self.func)

        return df

    def constraint(self, k: int) -> ca.MX:

        if k not in self.mx:
            raise ValueError(f'Did not find k={k} in the keys of mx for {self}.')

        if k not in self.base.mx:
            print(k, 'not in', self.base.mx.keys(), 'base:', self.base)
            raise ValueError(f'Did not find k={k} in the keys of mx for {self.base} which is base for {self}.')

        lhs = self[k]
        rhs = self.func(self.base[k])

        return lhs - rhs

    @property
    def past_steps(self) -> int:
        return 0
