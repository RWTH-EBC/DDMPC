from ddmpc import *

"""
This script is used to define the system, 
the relevant variables and what to plot during simulation
"""

time_offset = 1672527600    # unix time stamp: time offset to set the date to 01.01.2023 (0 is 01.01.1970)

# different modes for room air temperature
# defines boundaries, targets and time interval between different targets / set points
TAirRoom_random = Random(day_start=8, day_end=16, day_lb=273.15 + 19, night_lb=273.15 + 16, day_ub=273.15 + 21,
                         night_ub=273.15 + 24, interval=60 * 60 * 4)
TAirRoom_identification = Identification()
TAirRoom_economic = Economic(day_start=8, day_end=18, day_lb=273.15 + 21, day_ub=273.15 + 23, night_lb=273.15 + 17,
                             night_ub=273.15 + 28, weekend=True)

""" Define the features (Variables) of your system """
# creates room temperature [K] as Controlled object, later used in optimization function
# source is readable datapoint / FMU variable
# mode set as Identification (random sequence of targets within bounds with random intervals between targets)
TAirRoom = Controlled(  # The room temperature should be controlled
    source=Readable(
        name="Room Temperature",        # colloquial name
        read_name="TAirRoom",           # column name in df / name of datapoint or FMU variable
        plt_opts=red_line,              # here some customization for plotting
    ),
    mode=TAirRoom_identification,
)

# Change can calculate the change of in this case the room temperature between the current and the previous time step
TAirRoom_change = Connection(Change(base=TAirRoom))

# creates cold heat flow [W] as Tracking object
# Tracking objects only used to "measure" further variables / for evaluation purpose
# source is readable datapoint / FMU variable
Q_flowCold = Tracking(
    Readable(
        name="HeatFlow Cold",           # colloquial name
        read_name="QFlowCold",          # column name in df / name of datapoint or FMU variable
        plt_opts=blue_line,             # here some customization for plotting
    )
)

# creates hot heat flow [W] as Tracking object
# Tracking objects only used to "measure" further variables / for evaluation purpose
# source is readable datapoint / FMU variable
Q_flowHot = Tracking(
    Readable(
        name="HeatFlow Hot",            # colloquial name
        read_name="QFlowHeat",          # column name in df / name of datapoint or FMU variable
        plt_opts=red_line,              # here some customization for plotting
    )
)

# creates air temperature (input for AHU) [K] as Tracking object
# Tracking objects only used to "measure" further variables / for evaluation purpose
# source is readable datapoint / FMU variable
TAirIn = Tracking(
    Readable(
        name="AHU Measured",                # colloquial name
        read_name="Bus.ahuBus.TSupAirMea",  # column name in df / name of datapoint or FMU variable
        plt_opts=red_line,                  # here some customization for plotting
    )
)

# creates AHU set point temperature [K] as Control object
# control variables are manipulated by the controller
# source is readable datapoint / FMU variable
TsetAHU = Control(
    source=Readable(
        name="AHU SetPoint",                # colloquial name
        read_name="TAhuSet",                # column name in df / name of datapoint or FMU variable
        plt_opts=red_line,                  # here some customization for plotting
    ),
    lb=273.15 + 17,                         # lower bound
    ub=273.15 + 28,                         # upper bound
    default=273.15 + 22,                    # without controller the default value is active
)

# Change can calculate the change between the current and the previous time step
TsetAHU_change = Connection(
    Change(base=TsetAHU)
)  # later we want to penalize the change to prevent oscillations

# creates temperatures t_1 to t_4 [K] as Tracking objects
# Tracking objects only used to "measure" further variables / for evaluation purpose
# source is readable datapoint / FMU variable
t_1 = Tracking(
    Readable(
        name="AHU in hot",                                          # heater supply temperature; colloquial name
        read_name="Bus.ahuBus.heaterBus.hydraulicBus.TFwrdInMea",   # column name in df / name of datapoint or FMU variable
        plt_opts=black_line,                                        # here some customization for plotting
    )
)
t_2 = Tracking(
    Readable(
        name="AHU out hot",                                         # heater return temperature; colloquial name
        read_name="Bus.ahuBus.heaterBus.hydraulicBus.TRtrnOutMea",  # column name in df / name of datapoint or FMU variable
        plt_opts=black_line,                                        # here some customization for plotting
    )
)
t_3 = Tracking(
    Readable(
        name="AHU in cold",                                         # cooler supply temperature; colloquial name
        read_name="Bus.ahuBus.coolerBus.hydraulicBus.TFwrdInMea",   # column name in df / name of datapoint or FMU variable
        plt_opts=black_line,                                        # here some customization for plotting
    )
)
t_4 = Tracking(
    Readable(
        name="AHU out cold",                                        # cooler return temperature; colloquial name
        read_name="Bus.ahuBus.coolerBus.hydraulicBus.TRtrnOutMea",  # column name in df / name of datapoint or FMU variable
        plt_opts=black_line,                                        # here some customization for plotting
    )
)

# creates mass flow through heater / cooler [m^3/s] as Tracking object
# Tracking objects only used to "measure" further variables / for evaluation purpose
# source is readable datapoint / FMU variable
mass_flow_hot = Tracking(
    Readable(
        name="AHU MassFlow Hot",                                    # colloquial name
        read_name="Bus.ahuBus.heaterBus.hydraulicBus.VFlowInMea",   # column name in df / name of datapoint or FMU variable
        plt_opts=red_line,                                          # here some customization for plotting
    )
)
mass_flow_cold = Tracking(
    Readable(
        name="AHU MassFlow Cold",                                   # colloquial name
        read_name="Bus.ahuBus.coolerBus.hydraulicBus.VFlowInMea",   # column name in df / name of datapoint or FMU variable
        plt_opts=blue_line,                                         # here some customization for plotting
    )
)

# creates heat flow through heater / cooler [kW] as Tracking object
# Tracking objects only used to "measure" further variables / for evaluation purpose
# source is readable datapoint / FMU variable
heat_flow_hot = Tracking(
    HeatFlow(
        name="Heat Flow Hot",                                       # column name in df
        mass_flow=mass_flow_hot,
        temperature_in=t_1,
        temperature_out=t_2,
        plt_opts=red_line,                                          # here some customization for plotting
    )
)
heat_flow_cold = Tracking(
    HeatFlow(
        name="Heat Flow Cold",                                      # column name in df
        mass_flow=mass_flow_cold,
        temperature_in=t_3,
        temperature_out=t_4,
        plt_opts=blue_line,                                         # here some customization for plotting
    )
)

# creates total heat flow through AHU [kW] as Controlled object, later used in optimization function
# EnergyBalance provides methods to sum up the hot and cold heat flow
# mode set as Steady with 0 as target since heat flow / energy use should be as low as possible
Q_flowAhu = Controlled(
    EnergyBalance(
        name="AHU EnergyBalance",                                   # column name in df
        heat_flows=[heat_flow_cold, heat_flow_hot]
    ),
    mode=Steady(day_target=0, night_target=0),
)

# creates Concrete Core Activation (TABS) heat flow set point [kW] as Control object
# control variables are manipulated by the controller
# source is readable datapoint / FMU variable
Q_flowTabs = Control(
    Readable(
        name="Heat Flow SetPoint",                          # colloquial name
        read_name="QFlowTabsSet",                           # column name in df / name of datapoint or FMU variable
        plt_opts=red_line,                                  # here some customization for plotting
    ),
    lb=-5,
    ub=5,
    default=0,
)

# Change can calculate the change between the current and the previous time step
Q_flowTabs_change = Connection(Change(base=Q_flowTabs))

# creates ambient temperature [K] as Disturbance based on forecast
dry_bul = Disturbance(
    Readable(
        name="Ambient Temperature",                         # colloquial name
        read_name="weaBus.TDryBul",                         # column name in df / name of datapoint or FMU variable
        plt_opts=light_red_line,                            # here some customization for plotting
    )
)
# creates direct radiation [W/m^2] as Disturbance based on forecast
rad_dir = Disturbance(
    Readable(
        name="Dir. Rad.",                                   # colloquial name
        read_name="weaBus.HDirNor",                         # column name in df / name of datapoint or FMU variable
        plt_opts=red_line,                                  # here some customization for plotting
    )
)


# Define additional constructed features
def sin_d(t):
    return np.sin(2 * np.pi * t / 86400)        # 86400 seconds = 1 day


def cos_d(t):
    return np.cos(2 * np.pi * t / 86400)        # 86400 seconds = 1 day


def sin_w(t):
    return np.sin(2 * np.pi * t / 604800)       # 604800 seconds = 1 week


def cos_w(t):
    return np.cos(2 * np.pi * t / 604800)       # 604800 seconds = 1 week


# here the daytime and day of the week encoded as sin/cos are used to learn user behavior
daily_sin = Disturbance(TimeFunc(name="daily_sin", func=sin_d))
daily_cos = Disturbance(TimeFunc(name="daily_cos", func=cos_d))
weekly_sin = Disturbance(TimeFunc(name="weekly_sin", func=sin_w))
weekly_cos = Disturbance(TimeFunc(name="weekly_cos", func=cos_w))


""" Define the controlled system """
model = Model(*Feature.all)  # Create a model and pass all Features to it

system = FMU(
    model=model,
    step_size=one_minute * 15,                          # time between control steps
    name="ashrae140_900_set_point_fmu.fmu",             # file in \Examples\FMUs\ashrae\stored_data\FMUs
    time_offset=time_offset,
)  # initialize system


""" Define the Inputs and Outputs of the
 process models using the Training data class"""
# Define training data for supervised machine learning
# Room air temperature is controlled variable
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
    step_size=one_minute * 15,
)

# Define training data for supervised machine learning
# AHU heat flow set point is controlled variable
Q_flowAhu_TrainingData = TrainingData(
    inputs=Inputs(
        Input(dry_bul, lag=1),
        Input(TsetAHU, lag=1),
        Input(TAirRoom, lag=1),
    ),
    output=Output(source=Q_flowAhu),
    step_size=one_minute * 15,
)

""" Define which quantities should be plotted """
# Define plot / plot appearance for PID
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

# Define plot / plot appearance for MPC
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
