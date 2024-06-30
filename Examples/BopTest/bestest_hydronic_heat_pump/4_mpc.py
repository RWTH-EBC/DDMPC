from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *

mpc_name = 'test'
scenario = 'peak_heat_day'

# regression for TAirRoom
# TAirRoom_pred: LinearRegression = load_LinearRegression(filename='TairRoom_linReg')
# TAirRoom_ped: GaussianProcess = load_GaussianProcess(filename='TairRoom_GPR')

# load best NN trained before from disc
TAirRoom_pred: NeuralNetwork = load_NetworkTrainer(filename="TairRoom_ANN").best

# Define white box model for power of heat pump
power_hp_wb = WhiteBox(
    inputs=[u_hp.source, t_amb.source, TAirRoom.source, u_hp_logistic.source],
    output=power_hp,
    output_expression=(                         # prediction function
        u_hp.source
        * 10000
        / ((TAirRoom.source + 15) / ((TAirRoom.source + 15) - t_amb.source) * 0.6)
        + 1110 * u_hp_logistic.source
    ),
    step_size=one_minute * 15,
)
# power_hp_gpr = load_GaussianProcess("power_hp_GPR_500_IP")

TAirRoom.mode = TAirRoom_economic                   # changes mode previously defined in configuration.py
FileManager.experiment = f'{mpc_name}'              # changes path data will be saved to from now on

# Define MPC
hhp_MPC = ModelPredictive(
    step_size=one_minute * 15,              # step size of controller
    nlp=NLP(                                # non linear problem
        model=model,
        N=32,                               # prediction horizon
        objectives=[                        # objective function
            Objective(feature=TAirRoom, cost=Quadratic(weight=10)),
            Objective(feature=costs_el, cost=AbsoluteLinear(weight=1)),
            # Objective(feature=u_hp,             cost=Linear(1)),
            Objective(feature=u_hp_change, cost=Quadratic(weight=0.1)),
        ],
        constraints=[
            Constraint(feature=u_hp, lb=0, ub=1),
        ],
    ),
    forecast_callback=system.get_forecast,
    solution_plotter=mpc_solution_plotter,
    show_solution_plot=False,
    save_solution_plot=False,
    save_solution_data=True,
)

# set up the system
# if no scenario is given, given start_time and warmup_period are used to initialize the system
# otherwise the system is initialized based on the scenario-parameters (predefined in BOPTEST framework)
system.setup(
    start_time=0,
    warmup_period=one_week,
    scenario={'electricity_price': 'dynamic',
              'time_period': scenario},
    active_control_layers={"oveHeaPumY_activate": 1},
)

solver_options = {
    "verbose": False,
    "ipopt.print_level": 2,
    "ipopt.max_iter": 1000,
    "expand": True,
    'ipopt.tol': 1e-1,
    'ipopt.acceptable_tol': 1,
}

df = None

for repetition in range(14):        # for 14 days (standard period in BOPTEST to ensure comparability)

    # build nonlinear problem
    # default algorithm: ipopt
    hhp_MPC.nlp.build(
        solver_options=solver_options, predictors=[TAirRoom_pred, power_hp_wb]
    )

    # runs the system for the given duration using the given controller
    # duration has to be dividable by step size of the System
    # returns data frame (only current and not past data frames) in a DataContainer
    # plots data and saves plot to disk (directory: /stored_data/plots/[mpc_name]/ )
    online_data = system.run(controllers=(hhp_MPC,), duration=one_day * 1)
    online_data.plot(plotter=mpc_solution_plotter, save_plot=True, save_name=f'mpc_{repetition}.png')

    # # online learning TAirRoom
    # TAirRoom_TrainingData.add(online_data)
    # TAirRoom_TrainingData.shuffle()
    # TAirRoom_TrainingData.split(0.7, 0.15, 0.15)
    # TAirRoom_pred.fit(
    #     training_data=TAirRoom_TrainingData,
    #     epochs=100,
    #     batch_size=50,
    #     verbose=1,
    # )
    # TAirRoom_pred.test(training_data=TAirRoom_TrainingData, show_plot=True)
    # power_hp_TrainingData.add(online_data)
    # power_hp_TrainingData.shuffle()
    # power_hp_TrainingData.split(0.8, 0, 0.2)
    # power_hp_gpr.fit(training_data=power_hp_TrainingData)
    # power_hp_gpr.test(training_data=power_hp_TrainingData)

    # concat data frame of current repetition to data frame of previous iterations if existing
    if df is None:
        df = online_data.df
    else:
        df = pd.concat([df, online_data.df], axis=0)

# save data frame with data from all repetitions to file data.csv (directory: /stored_data/[mpc_name]/ )
df.to_csv(str(Path(FileManager.experiment_dir(), 'data.csv')))

# obtain / calculate kpis from system (calculated from start_time, not including warm_up period)
# put kpis in data frame and save this to file kpis.csv (directory: /stored_data/[mpc_name]/ )
kpis = system.get_kpis()
kpis = pd.DataFrame(data=kpis, index=[0])
kpis.to_csv(str(Path(FileManager.experiment_dir(), 'kpis.csv')), index=False)
