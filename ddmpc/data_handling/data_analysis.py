import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn import linear_model

from ddmpc.data_handling.storing_data import *


def compare(
        dh: DataHandler,
        f1,
        f2,
        c:          Feature = None,
        cmap:       LinearSegmentedColormap = None,
        f1_label:   str = None,
        f2_label:   str = None,
        c_label:    str = None,
        title:      str = None,
        save:       bool = False,
        filepath:   str = None,
        size:       tuple = (5, 5),
):

    fig = matplotlib.pyplot.gcf()

    if f1_label is None:
        f1_label = f1.source.name
    if f2_label is None:
        f2_label = f2.source.name

    data = list()

    for dc in dh:
        data.append(dc.df)

    col1 = f1.source.col_name
    col2 = f2.source.col_name
    plt.xlabel(f1_label)
    plt.ylabel(f2_label)

    data = pd.concat(data)
    if c is None:
        data = data[[col1, col2]]
    else:
        col3 = c.source.col_name
        data = data[[col1, col2, col3]]

    data.dropna(inplace=True)

    x = data[col1].values.reshape(-1, 1)
    y = data[col2].values.reshape(-1, 1)
    if c is None:
        c = fmt.dark_grey
    else:
        c = data[c.source.col_name]

    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    print(f'Linear Regression coefficient of {f1.source.name} and {f2.source.name} = {regr.coef_}')

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(name='red', colors=[fmt.grey, fmt.red])

    if c_label is None:
        plt.scatter(x, y, c=c, s=1, cmap=cmap)
    else:
        plt.scatter(x, y, c=c, s=1, cmap=cmap, label=c_label)
        plt.legend(loc='upper right')

    plt.plot(x, regr.predict(x), color=fmt.black, linewidth=1, linestyle='dotted')

    if cmap is not None:
        norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm)

    fig.set_size_inches(size)
    # plt.subplots_adjust(bottom=0.18)

    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(filepath)

    plt.show()


def load_solutions(file: str) -> tuple[dict[float, pd.DataFrame], float, float]:

    solutions = read_pkl(file)

    if not isinstance(solutions, dict):
        raise TypeError()

    for time, solution in solutions.items():
        if not isinstance(solution, pd.DataFrame):
            raise TypeError()
        """
        else:
            if True:
                print(time, len(solution.index))
        """

    return solutions, list(solutions.keys())[0], list(solutions.keys())[-1]


def load_history(file: str) -> pd.DataFrame:

    history = read_pkl(file)

    return history
