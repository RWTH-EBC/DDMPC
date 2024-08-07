from Examples.FMUs.ashrae.config import *

mpc_name = 'test'
FileManager.experiment = f'{mpc_name}'      # changes path data will be saved to from now on

""" Choose the process models """
# # load best NN trained before from disc
# TAirRoom_predictor: NeuralNetwork = load_NetworkTrainer(filename="TAirRoom_ann").best
# Q_flowAhu_predictor: NeuralNetwork = load_NetworkTrainer(filename="Q_flowAhu_ann").best

# TAirRoom_predictor: GaussianProcess = load_GaussianProcess('TAirRoom_GPR')
# Q_flowAhu_predictor: GaussianProcess = load_GaussianProcess("Q_flowAhu_GPR")

TAirRoom_predictor: LinearRegression = load_LinearRegression("TAirRoom_linreg")
Q_flowAhu_predictor: LinearRegression = load_LinearRegression("Q_flowAhu_linreg")


TAirRoom.mode = TAirRoom_economic           # changes mode previously defined in config.py

""" Initialize Model Predictive Controller """
ThermalZone_MPC = ModelPredictive(
    step_size=one_minute * 15,              # step size of controller
    nlp=NLP(                                # non linear problem
        model=model,
        N=32,                               # prediction horizon
        objectives=[                        # objective function
            Objective(feature=TAirRoom, cost=Quadratic(weight=100)),
            Objective(feature=Q_flowTabs, cost=AbsoluteLinear(0.5)),
            Objective(feature=Q_flowAhu, cost=AbsoluteLinear(1)),
            Objective(feature=Q_flowTabs_change, cost=Quadratic(0.1)),
            Objective(feature=TsetAHU_change, cost=Quadratic(0.1)),
        ],
        constraints=[
            Constraint(feature=Q_flowTabs, lb=-5, ub=5),
            Constraint(feature=TsetAHU, lb=273.15 + 17, ub=273.15 + 25),
        ],
    ),
    forecast_callback=system.get_forecast,
    solution_plotter=mpc_solution_plotter,
    show_solution_plot=True,
    save_solution_plot=False,
    save_solution_data=True,
)

# set up the system
# here: default fmu instance and simulation tolerance used
system.setup(start_time=0)

# run the simulation for one day without controller to settle the system
system.run(duration=one_day)

solver_options = {
    "verbose": False,
    "ipopt.print_level": 2,
    "ipopt.max_iter": 1000,
    "expand": True,
    'ipopt.tol': 1e-1,
    'ipopt.acceptable_tol': 1,
}

# # Load initial data to consider for retraining
# dh = load_DataHandler("pid_data")

df = None

"""  Online learning loop """
for repetition in range(14):  # for 14 days

    # build nonlinear problem with (re-)trained models
    # default algorithm: ipopt
    ThermalZone_MPC.nlp.build(
        solver_options=solver_options,
        predictors=[Q_flowAhu_predictor, TAirRoom_predictor],
    )

    # print a summary of the NLP
    ThermalZone_MPC.nlp.summary()

    # runs the system for the given duration using the given MPC controller
    # duration has to be dividable by step size of the system
    # returns data frame (only current and not past data frames) in a DataContainer
    # plots data and saves plot to disk (directory: /stored_data/plots/[mpc_name]/ )
    online_data = system.run(controllers=[ThermalZone_MPC], duration=one_day * 1)
    online_data.plot(plotter=mpc_solution_plotter, save_plot=True, save_name=f'mpc_{repetition}.png')

    # # online learning TAirRoom
    # TAirRoom_TrainingData.clear()
    # TAirRoom_TrainingData.add(online_data)
    # TAirRoom_TrainingData.shuffle()
    # TAirRoom_TrainingData.split(0.7, 0.15, 0.15)
    # TAirRoom_predictor.fit(  # Arguments of the fit function may vary depending on model type
    #     training_data=TAirRoom_TrainingData,
    #     # epochs=50,
    #     # batch_size=20,
    #     # verbose=1,
    # )
    # TAirRoom_predictor.test(training_data=TAirRoom_TrainingData)
    # Q_flowAhu_TrainingData.clear()
    # Q_flowAhu_TrainingData.add(online_data)
    # Q_flowAhu_TrainingData.shuffle()
    # Q_flowAhu_TrainingData.split(0.8, 0.0, 0.2)
    # Q_flowAhu_predictor.fit(training_data=Q_flowAhu_TrainingData)
    # Q_flowAhu_predictor.test(training_data=Q_flowAhu_TrainingData)

    # concat data frame of current repetition to data frame of previous iterations if existing
    if df is None:
        df = online_data.df
    else:
        df = pd.concat([df, online_data.df], axis=0)
system.close()

# save data frame with data from all repetitions to file data.csv (directory: /stored_data/[mpc_name]/ )
df.to_csv(str(Path(FileManager.experiment_dir(), 'data.csv')))
