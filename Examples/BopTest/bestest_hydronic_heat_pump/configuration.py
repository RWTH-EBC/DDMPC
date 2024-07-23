from ddmpc import *

"""
This script is used to define the system, 
the relevant variables and what to plot during simulation
"""

time_offset = 1546300800    # unix time stamp: time offset to set the date to 01.01.2019 (0 is 01.01.1970)

# different modes for room air temperature
# defines boundaries, targets and time interval between different targets / set points
TAirRoom_steady = Steady(day_start=8, day_end=15, day_target=290.15, night_target=294.15)
TAirRoom_random = Random(day_start=8, day_end=19, day_lb=288.15, day_ub=303.15, night_lb=294.15, night_ub=297.15,
                         interval=3600 * 6)
TAirRoom_economic = Economic(day_start=8, day_end=19, day_lb=288.15, day_ub=303.15, night_lb=294.15, night_ub=297.15)

""" Define the features (Variables) of your system """
# creates room temperature [K] as Controlled object, later used in optimization function
# an output given through BOPTEST framework is used as source
# mode set as steady
# read name and documentation given in BOPTEST framework on
# https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/index.html
TAirRoom = Controlled(
    source=Readable(
        name="TAir",                            # colloquial name
        read_name="reaTZon_y",                  # zone operative temperature; column name in df
        plt_opts=PlotOptions(color=red, line=line_solid),
    ),
    mode=TAirRoom_steady,
)

# Change can calculate the change of in this case the room temperature between the current and the previous time step
TAirRoom_change = Connection(Change(base=TAirRoom))

# creates ambient temperature [K] as Disturbance based on forecast given through BOPTEST framework
# read name, forecast name and documentation given in BOPTEST framework on
# https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/index.html
t_amb = Disturbance(
    Readable(
        name="Ambient temperature",             # colloquial name
        read_name="weaSta_reaWeaTDryBul_y",     # outside dry bulb temperature measurement; column name in df
        plt_opts=PlotOptions(color=blue, line=line_solid, label="Ambient Temperature"),
    ),
    forecast_name="TDryBul",                    # dry bulb temperature at ground level [K]
)

# creates heat pump modulating signal [1] as Control object
# control variables are manipulated by the controller
# read name and documentation (e.g. configuration boundaries) given in BOPTEST framework on
# https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/index.html
u_hp = Control(
    source=Readable(
        name="u_hp",                            # colloquial name
        read_name="oveHeaPumY_u",               # heat pump modulating signal for compressor speed; column name in df
        plt_opts=PlotOptions(color=blue, line=line_solid, label="u_hp"),
    ),
    lb=0,                                       # not working
    ub=1,                                       # working at maximum capacity
    default=0,
    cutoff=0.1,                                 # everything below cutoff results in signal 0 due to minimal rotational speed of hp
)

# creates direct radiation [W/m^2] as Disturbance based on forecast given through BOPTEST framework
# read name, forecast name and documentation given in BOPTEST framework on
# https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/index.html
rad_dir = Disturbance(
    Readable(
        name="direct radiation",                # colloquial name
        read_name="weaSta_reaWeaHDirNor_y",     # direct normal radiation measurement; column name in df
        plt_opts=PlotOptions(color=light_red, line=line_solid, label="Radiation"),
    ),
    forecast_name="HDirNor",                    # direct normal radiation
)

# Change can calculate the change between the current and the previous time step
t_amb_change = Connection(Change(base=t_amb))
rad_dir_change = Connection(Change(base=rad_dir))

# creates power of fan [W] as Tracking object
# Tracking objects only used to "measure" further variables / for evaluation purpose
# read name and documentation given in BOPTEST framework on
# https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/index.html
power_fan = Tracking(
    Readable(
        name="el. Power Fan",                   # colloquial name
        read_name="reaPFan_y",                  # electrical power of the heat pump evaporator fan; column name in df
        plt_opts=PlotOptions(color=blue, line=line_dotted, label="P_fan"),
    )
)

# creates power of heat pump [W] as Controlled object, later used in optimization function
# read name and documentation given in BOPTEST framework on
# https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/index.html
power_hp = Controlled(
    source=Readable(
        name="el. Power HP",                    # colloquial name
        read_name="reaPHeaPum_y",               # heat pump electrical power; column name in df
        plt_opts=PlotOptions(color=red, line=line_solid, label="P_hp"),
    ),
    mode=Steady(day_target=0, night_target=0),  # power of heat pump should be as low as possible
)

# creates power of emission circuit pump [W] as Tracking object
# Tracking objects only used to "measure" further variables / for evaluation purpose
# read name and documentation given in BOPTEST framework on
# https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/index.html
power_ec = Tracking(
    Readable(
        name="el. Power emission circuit",      # colloquial name
        read_name="reaPPumEmi_y",               # emission circuit pump electrical power; column name in df
        plt_opts=PlotOptions(color=grey, line=line_dashdot, label="P_ec"),
    )
)

# creates evaporator fan signal [1] as Tracking object
# Tracking objects only used to "measure" further variables / for evaluation purpose
# read name and documentation given in BOPTEST framework on
# https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/index.html
u_fan = Tracking(
    source=Readable(
        name="u_fan",                           # colloquial name
        read_name="oveFan_u",                   # signal to control the heat pump evaporator fan (either on or off); column name in df
        plt_opts=PlotOptions(color=blue, line=line_solid, label="u_fan"),
    ),
)

# Change can calculate the change between the current and the previous time step
u_hp_change = Connection(Change(base=u_hp))
u_fan_change = Tracking(Change(base=u_fan))

# creates electricity price [Euro/kWh] as Disturbance based on forecast given through BOPTEST framework
# read name and documentation given in BOPTEST framework on
# https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/index.html
price_el = Disturbance(
    Readable(
        name="el. Power Price",                 # colloquial name
        read_name="PriceElectricPowerDynamic",  # Electricity price for a day / night tariff; column name in df
        plt_opts=PlotOptions(color=light_red, line=line_solid, label="Price"),
    ),
    forecast_name="PriceElectricPowerDynamic",
)

# Product can calculate the product between two values at a single or at every time step
costs_el = Connection(Product(b1=price_el, b2=power_hp, scale=0.001))


def logistic(x):
    return 1 / (1 + np.exp(-(x - 0.01) * 500))


# Func can apply given function to u_hp
u_hp_logistic = Connection(Func(base=u_hp, func=logistic, name="logistic"))


""" Define the controlled system """
model = Model(*Feature.all)         # Create a model and pass all Features to it

system = BopTest(
    model=model,
    step_size=one_minute * 15,              # time between control steps
    url="http://127.0.0.1:5000/",     # url of server with BOPTEST framework
    time_offset=time_offset,
)  # initialize system


""" Define the Inputs and Outputs of the
 process models using the Training data class"""
# Define training data for supervised machine learning (power_hp)
# power of heat pump is controlled variable
power_hp_TrainingData = TrainingData(
    inputs=Inputs(
        Input(source=u_hp, lag=1),
        Input(source=u_hp_logistic, lag=1),
        Input(source=t_amb, lag=1),
        Input(source=TAirRoom, lag=1),
    ),
    output=Output(power_hp),
    step_size=one_minute * 15,
)

# Define training data for supervised machine learning
# Room air temperature is controlled variable
TAirRoom_TrainingData = TrainingData(
    inputs=Inputs(
        Input(source=TAirRoom, lag=3),
        Input(source=t_amb, lag=2),
        Input(source=rad_dir, lag=1),
        Input(source=u_hp, lag=3),
    ),
    output=Output(TAirRoom_change),
    step_size=one_minute * 15,
)


""" Define which quantities should be plotted """
# Define plot / plot appearance for PID
pid_plotter = Plotter(
    SubPlot(features=[TAirRoom], y_label="Room temperature in °C", shift=273.15),
    SubPlot(features=[u_hp], y_label="Modulation hp", step=True),
    SubPlot(features=[u_fan], y_label="Modulation fan", step=True),
    SubPlot(features=[t_amb, rad_dir], y_label="disturbances in %", normalize=True),
    SubPlot(features=[power_hp], y_label="el. Power in W", normalize=False),
    SubPlot(features=[costs_el], y_label="el. Costs in ct")
)

# Define plot / plot appearance for MPC
mpc_plotter = Plotter(
    SubPlot(features=[TAirRoom], y_label="Room temperature in °C", shift=273.15),
    SubPlot(features=[u_hp], y_label="Modulation hp", step=True),
    SubPlot(features=[u_fan], y_label="Modulation fan", step=True),
    SubPlot(features=[t_amb, rad_dir], y_label="disturbances in %", normalize=True),
    SubPlot(features=[power_hp], y_label="el. Power in W", step=True, normalize=False),
    SubPlot(features=[costs_el], y_label="el. Costs in ct"),
    SubPlot(features=[price_el], y_label="el. price in €/kW")
)

# Define plot / plot appearance for MPC solution
mpc_solution_plotter = Plotter(
    SubPlot(features=[TAirRoom], y_label="Room temperature in °C", shift=273.15),
    SubPlot(features=[u_hp], y_label="Modulation hp", step=True),
    SubPlot(features=[t_amb], y_label="Ambient temperature in °C"),
    SubPlot(features=[rad_dir], y_label="Radiation"),
    SubPlot(features=[power_hp], y_label="el. Power in W", step=True, normalize=False),
    SubPlot(features=[costs_el], y_label="el. Costs in €"),
    SubPlot(features=[price_el], y_label="el. price in €/kW")
)
