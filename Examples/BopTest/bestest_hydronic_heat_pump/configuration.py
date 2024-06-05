from ddmpc import *

time_offset = 1546300800

TAirRoom_steady = Steady(day_start=8, day_end=15, day_target=290.15, night_target=294.15)
TAirRoom_random = Random(day_start=8, day_end=19, day_lb=288.15, day_ub=303.15, night_lb=294.15, night_ub=297.15,
                         interval=3600 * 6)
TAirRoom_economic = Economic(day_start=8, day_end=19, day_lb=288.15, day_ub=303.15, night_lb=294.15, night_ub=297.15)

TAirRoom = Controlled(
    source=Readable(
        name="TAir",
        read_name="reaTZon_y",
        plt_opts=PlotOptions(color=red, line=line_solid),
    ),
    mode=TAirRoom_steady,
)

TAirRoom_change = Connection(Change(base=TAirRoom))

t_amb = Disturbance(
    Readable(
        name="Ambient temperature",
        read_name="weaSta_reaWeaTDryBul_y",
        plt_opts=PlotOptions(color=blue, line=line_solid, label="Ambient Temperature"),
    ),
    forecast_name="TDryBul",
)

u_hp = Control(
    source=Readable(
        name="u_hp",
        read_name="oveHeaPumY_u",
        plt_opts=PlotOptions(color=blue, line=line_solid, label="u_hp"),
    ),
    lb=0,
    ub=1,
    default=0,
    cutoff=0.1,
)

rad_dir = Disturbance(
    Readable(
        name="direct radiation",
        read_name="weaSta_reaWeaHDirNor_y",
        plt_opts=PlotOptions(color=light_red, line=line_solid, label="Radiation"),
    ),
    forecast_name="HDirNor",
)


t_amb_change = Connection(Change(base=t_amb))
rad_dir_change = Connection(Change(base=rad_dir))

power_fan = Tracking(
    Readable(
        name="el. Power Fan",
        read_name="reaPFan_y",
        plt_opts=PlotOptions(color=blue, line=line_dotted, label="P_fan"),
    )
)

power_hp = Controlled(
    source=Readable(
        read_name="reaPHeaPum_y",
        name="el. Power HP",
        plt_opts=PlotOptions(color=red, line=line_solid, label="P_hp"),
    ),
    mode=Steady(day_target=0, night_target=0),
)


power_ec = Tracking(
    Readable(
        name="el. Power emission circuit",
        read_name="reaPPumEmi_y",
        plt_opts=PlotOptions(color=grey, line=line_dashdot, label="P_ec"),
    )
)


u_fan = Tracking(
    source=Readable(
        name="u_fan",
        read_name="oveFan_u",
        plt_opts=PlotOptions(color=blue, line=line_solid, label="u_fan"),
    ),
)

u_hp_change = Connection(Change(base=u_hp))
u_fan_change = Tracking(Change(base=u_fan))

price_el = Disturbance(
    Readable(
        name="el. Power Price",
        read_name="PriceElectricPowerDynamic",
        plt_opts=PlotOptions(color=light_red, line=line_solid, label="Price"),
    ),
    forecast_name="PriceElectricPowerDynamic",
)

costs_el = Connection(Product(b1=price_el, b2=power_hp, scale=0.001))

def logistic(x):
    return 1 / (1 + np.exp(-(x - 0.01) * 500))


u_hp_logistic = Connection(Func(base=u_hp, func=logistic, name="logistic"))

model = Model(*Feature.all)

system = BopTest(
    model=model,
    step_size=one_minute * 15,
    url="http://127.0.0.1:5000",
    time_offset=time_offset,
)

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


pid_plotter = Plotter(
    SubPlot(features=[TAirRoom], y_label="Room temperature in °C", shift=273.15),
    SubPlot(features=[u_hp], y_label="Modulation hp", step=True),
    SubPlot(features=[u_fan], y_label="Modulation fan", step=True),
    SubPlot(features=[t_amb, rad_dir], y_label="disturbances in %", normalize=True),
    SubPlot(features=[power_hp], y_label="el. Power in W", normalize=False),
    SubPlot(features=[costs_el], y_label="el. Costs in ct")
)

mpc_plotter = Plotter(
    SubPlot(features=[TAirRoom], y_label="Room temperature in °C", shift=273.15),
    SubPlot(features=[u_hp], y_label="Modulation hp", step=True),
    SubPlot(features=[u_fan], y_label="Modulation fan", step=True),
    SubPlot(features=[t_amb, rad_dir], y_label="disturbances in %", normalize=True),
    SubPlot(features=[power_hp], y_label="el. Power in W", step=True, normalize=False),
    SubPlot(features=[costs_el], y_label="el. Costs in ct"),
    SubPlot(features=[price_el], y_label="el. price in €/kW")
)

mpc_solution_plotter = Plotter(
    SubPlot(features=[TAirRoom], y_label="Room temperature in °C", shift=273.15),
    SubPlot(features=[u_hp], y_label="Modulation hp", step=True),
    SubPlot(features=[t_amb], y_label="Ambient temperature in °C"),
    SubPlot(features=[rad_dir], y_label="Radiation"),
    SubPlot(features=[power_hp], y_label="el. Power in W", step=True, normalize=False),
    SubPlot(features=[costs_el], y_label="el. Costs in €"),
    SubPlot(features=[price_el], y_label="el. price in €/kW")
)
