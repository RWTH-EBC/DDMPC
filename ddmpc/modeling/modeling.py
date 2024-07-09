import ddmpc.utils.formatting as fmt
from ddmpc.modeling.features import *


class Model:
    """
    provides methods to
        - validate the model
        - print a summary of the model
        - update features at a single time step
        - update features at all time steps
    """

    def __init__(self, *features: Feature):
        """
        When model is instantiated, the features are sorted in a way that all subs are sorted with their parents
        Furthermore, the model is validated.
        """
        self.features = features

        self._sort()

        self._source_mapping: dict[Source, Feature] = dict()

        for f in self.features:
            self._source_mapping[f.source] = f

        self._validate()

    def __str__(self) -> str:
        return 'Model'

    def __repr__(self):
        return 'Model'

    def __getitem__(self, source: Source) -> Feature:
        """ Returns the Feature to the given Source """

        assert isinstance(source, Source), f'{source} is not of type Source.'

        return self._source_mapping[source]

    def summary(self):
        """
        prints a summary of the model.
        Tables of Controlled, Controls, Disturbances, Connection and Tracking objects
        each displayed with its key attributes (e.g. name, subs, mode, bounds)
        """

        print(f'{fmt.BOLD}Model Summary:{fmt.ENDC}')
        rows = [['Controlled  ', 'name', 'subs', 'mode']]

        for f in self.controlled:
            rows.append(['', f.source.name, f.mode, f.source.subs])

        rows.append([''])

        rows.append(['Controls    ', 'name', 'lb', 'ub', 'default', 'subs'])

        for f in self.controls:
            rows.append(['', f.source.name, f.lb, f.ub, f.default, f.source.subs])

        rows.append([''])

        rows.append(['Disturbances', 'name', 'subs', 'forecast_name'])

        for f in self.disturbances:
            rows.append(['', f.source.name, f.forecast_name, f.source.subs])

        rows.append([''])

        rows.append(['Connecting', 'name', 'subs'])

        for f in self.connecting:
            rows.append(['', f.source.name, f.source.subs])

        rows.append([''])

        rows.append(['Ignored', 'name', 'subs'])

        for f in self.ignored:
            rows.append(['', f.source.name, f.source.subs])

        fmt.print_table(rows)

    def _sort(self):
        """ sorts the sources of the model in a way, that all subs are sorted after their parents """

        lst = list(self.features)
        n = len(lst)

        def iteration() -> bool:

            for i in range(n):

                f1: Feature = lst[i]

                for j in range(i + 1, n):

                    f2: Feature = lst[j]

                    if f2.source in f1.source.subs:

                        lst[i], lst[j] = lst[j], lst[i]
                        return True

            return False

        while iteration():
            pass

        self.features = tuple(lst)

    def _validate(self):
        """ Validates the model structure and completion """

        # check for only features
        for f in self.features:
            assert isinstance(f, Feature), f'{f} is not a Feature.'

        # checks for duplicates
        for i, f1 in enumerate(self.features):
            for j, f2 in enumerate(self.features[i+1:]):
                if f1 == f2:
                    raise ValueError(f'Duplicate found at position {i} and {i+j}! {f1, f2}, {self.features}')

        # sub feature check
        features_to_add = list()
        for feature in self.features:
            for sub in feature.source.subs:
                if isinstance(self[sub], Tracking):
                    if isinstance(feature, Connection):

                        raise ValueError(f"{sub} cant be of type Tracking, when {feature} is not.")

                if sub not in [f.source for f in self.features]:
                    if sub not in features_to_add:
                        features_to_add.append(sub)

        assert len(features_to_add) == 0, f'Please make sure to add these Features to the Model: {features_to_add}'

    @property
    def controlled(self) -> list[Controlled]:
        """returns a list of all Controlled objects in features"""
        return [f for f in self.features if isinstance(f, Controlled)]

    @property
    def controls(self) -> list[Control]:
        """returns a list of all Control objects in features"""
        return [f for f in self.features if isinstance(f, Control)]

    @property
    def disturbances(self) -> list[Disturbance]:
        """returns a list of all Disturbance objects in features"""
        return [f for f in self.features if isinstance(f, Disturbance)]

    @property
    def connecting(self) -> list[Connection]:
        """returns a list of all Connection objects in features"""
        return [f for f in self.features if isinstance(f, Connection)]

    @property
    def ignored(self) -> list[Tracking]:
        """returns a list of all Tracking objects in features"""
        return [f for f in self.features if isinstance(f, Tracking)]

    @property
    def constructed(self) -> list[Constructed]:
        """returns a list of all objects in features that have a Constructed object as a source"""
        return [f.source for f in self.features if isinstance(f.source, Constructed)]

    @property
    def readable(self) -> list[Readable]:
        """returns a list of all objects in features that have a Readable object as a source"""
        return [f.source for f in self.features if isinstance(f.source, Readable)]

    def update(self, idx: int, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """updates every feature of the model at a given index (row) and writes result to df"""
        for feature in self.features:

            df = feature.update(df=df, idx=idx, inplace=inplace)

        return df

    def process(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """updates every feature of the model (all rows) and writes result to df"""
        for feature in self.features:
            df = feature.process(df=df, inplace=inplace)

        return df
