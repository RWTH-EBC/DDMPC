from ddmpc import *

"""
This script is used to define the system, 
the relevant variables and what to plot during simulation
"""


""" Define the features (Variables) of your system """
TAirRoom = Controlled(  # The room temperature should be controlled
    source=Readable(
        name="Room Temperature",  # internal name
        read_name="TAirRoom",  # name of datapoint or FMU variable
        plt_opts=red_line,  # here some customization for plotting
    ),
    mode=Random(),  # Control mode (defines boundaries and control goal etc
)
TAirRoom_change = Connection(
    Change(base=TAirRoom)
)  # we want to predict the temperature change

Q_flowCold = Tracking(  # Tracked variables for evaluation purpose
    Readable(name="HeatFlow Cold", read_name="QFlowCold", plt_opts=blue_line)
)

Q_flowHot = Tracking(
    Readable(name="HeatFlow Hot", read_name="QFlowHeat", plt_opts=red_line)
)
TAirIn = Tracking(
    Readable(name="AHU Measured", read_name="Bus.ahuBus.TSupAirMea", plt_opts=red_line)
)

TsetAHU = Control(  # control variables are manipulated by the controller
    source=Readable(
        name="AHU SetPoint",
        read_name="TAhuSet",
        plt_opts=red_line,
    ),
    lb=273.15 + 17,
    ub=273.15 + 28,
    default=273.15 + 22,  # without controller the default value is active
)

TsetAHU_change = Connection(
    Change(base=TsetAHU)
)  # later we want to penalize the change to prevent oscillations

t_1 = Tracking(
    Readable(
        name="AHU in hot",
        read_name="Bus.ahuBus.heaterBus.hydraulicBus.TFwrdInMea",
        plt_opts=black_line,
    )
)
t_2 = Tracking(
    Readable(
        name="AHU out hot",
        read_name="Bus.ahuBus.heaterBus.hydraulicBus.TRtrnOutMea",
        plt_opts=black_line,
    )
)
t_3 = Tracking(
    Readable(
        name="AHU in cold",
        read_name="Bus.ahuBus.coolerBus.hydraulicBus.TFwrdInMea",
        plt_opts=black_line,
    )
)
t_4 = Tracking(
    Readable(
        name="AHU out cold",
        read_name="Bus.ahuBus.coolerBus.hydraulicBus.TRtrnOutMea",
        plt_opts=black_line,
    )
)
mass_flow_hot = Tracking(
    Readable(
        name="AHU MassFlow Hot",
        read_name="Bus.ahuBus.heaterBus.hydraulicBus.VFlowInMea",
        plt_opts=red_line,
    )
)
mass_flow_cold = Tracking(
    Readable(
        name="AHU MassFlow Cold",
        read_name="Bus.ahuBus.coolerBus.hydraulicBus.VFlowInMea",
        plt_opts=blue_line,
    )
)

heat_flow_hot = Tracking(
    HeatFlow(
        name="Heat Flow Hot",
        mass_flow=mass_flow_hot,
        temperature_in=t_1,
        temperature_out=t_2,
        plt_opts=red_line,
    )
)
heat_flow_cold = Tracking(
    HeatFlow(
        name="Heat Flow Cold",
        mass_flow=mass_flow_cold,
        temperature_in=t_3,
        temperature_out=t_4,
        plt_opts=blue_line,
    )
)

Q_flowAhu = Controlled(
    EnergyBalance(name="AHU EnergyBalance", heat_flows=[heat_flow_cold, heat_flow_hot]),
    mode=Steady(day_target=0, night_target=0),
)


Q_flowTabs = Control(
    Readable(
        name="Heat Flow SetPoint",
        read_name="QFlowTabsSet",
        plt_opts=red_line,
    ),
    lb=-5,
    ub=5,
    default=0,
)

Q_flowTabs_change = Connection(Change(base=Q_flowTabs))

dry_bul = Disturbance(  # for disturbances a forecast is needed
    Readable(
        name="Outside Temperature", read_name="weaBus.TDryBul", plt_opts=light_red_line
    )
)
rad_dir = Disturbance(
    Readable(name="Dir. Rad.", read_name="weaBus.HDirNor", plt_opts=red_line)
)


# Define additional constructed features
def sin_d(t):
    return np.sin(2 * np.pi * t / 86400)


def cos_d(t):
    return np.cos(2 * np.pi * t / 86400)


def sin_w(t):
    return np.sin(2 * np.pi * t / 604800)


def cos_w(t):
    return np.cos(2 * np.pi * t / 604800)


# here the daytime and day of the week encoded as sin/cos are used to learn user behavior
daily_sin = Disturbance(TimeFunc(name="daily_sin", func=sin_d))
daily_cos = Disturbance(TimeFunc(name="daily_cos", func=cos_d))
weekly_sin = Disturbance(TimeFunc(name="weekly_sin", func=sin_w))
weekly_cos = Disturbance(TimeFunc(name="weekly_cos", func=cos_w))

""" Define the controlled system """
model = Model(*Feature.all)  # pass all features to the model
system = FMU(
    model=model, step_size=60 * 15, name="ashrae140_900_set_point_fmu.fmu"
)  # initialize system

""" Define the Inputs and Outputs of the
 process models using the Training data class"""

TAirRoom_TrainingData = TrainingData(
    inputs=Inputs(
        Input(TAirRoom, lag=1),
        Input(Q_flowTabs, lag=3),  # the lags define the considered time steps
        Input(TsetAHU, lag=2),
        Input(dry_bul, lag=2),
        Input(rad_dir, lag=1),
        Input(daily_sin, lag=2),
        Input(daily_cos, lag=2),
        Input(weekly_sin, lag=2),
        Input(weekly_cos, lag=2),
    ),
    output=Output(source=TAirRoom_change),
    step_size=60 * 15,
)

Q_flowAhu_TrainingData = TrainingData(
    inputs=Inputs(
        Input(dry_bul, lag=1),
        Input(TsetAHU, lag=1),
        Input(TAirRoom, lag=1),
    ),
    output=Output(source=Q_flowAhu),
    step_size=60 * 15,
)

""" Define which quantities should be plotted """
pid_plotter = Plotter(
    SubPlot(features=[TAirRoom], y_label="Air Temperature", shift=273.15, legend=False),
    SubPlot(features=[Q_flowTabs], y_label="BKT SetPoint", step=True, legend=False),
    SubPlot(
        features=[TsetAHU],
        y_label="AHU SetPoint",
        shift=273.15,
        lb=14,
        ub=26,
        step=True,
        legend=False,
    ),
    SubPlot(features=[Q_flowAhu], y_label="AHU Q", step=True, legend=False),
)
mpc_plotter = Plotter(
    SubPlot(features=[TAirRoom], y_label="Air Temperature", shift=273.15, legend=False),
    SubPlot(features=[Q_flowTabs], y_label="BKT SetPoint", step=True, legend=False),
    SubPlot(
        features=[TsetAHU],
        y_label="AHU SetPoint",
        shift=273.15,
        lb=14,
        ub=26,
        step=True,
        legend=False,
    ),
    SubPlot(features=[Q_flowAhu], y_label="AHU Q", step=True, legend=False),
    SubPlot(features=[dry_bul, rad_dir], y_label="Dist.", normalize=True),
)
