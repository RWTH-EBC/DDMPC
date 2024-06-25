from Examples.FMUs.ashrae.config import *

""" Choose the process models """
# TAirRoom_predictor = load_NeuralNetwork("TAirRoom_ann")
# Q_flowAhu_predictor = load_NeuralNetwork("Q_flowAhu_ann")

# TAirRoom_predictor: GaussianProcess = load_GaussianProcess('TAirRoom_GPR')
# Q_flowAhu_predictor: GaussianProcess = load_GaussianProcess("Q_flowAhu_GPR")

TAirRoom_predictor: LinearRegression = load_LinearRegression("TAirRoom_linreg")
Q_flowAhu_predictor: LinearRegression = load_LinearRegression("Q_flowAhu_linreg")

""" Set the comfort boundaries """
TAirRoom.mode = Economic(
    day_start=8,  # Time to activate the daytime boundaries
    day_end=18,  # Time to activate the nighttime boundaries
    day_lb=273.15 + 21,
    day_ub=273.15 + 23,
    night_lb=273.15 + 17,
    night_ub=273.15 + 28,
    weekend=True,  # Considers the nighttime constraints for weekends if true
)

""" Initialize Model Predictive Controller """
ThermalZone_MPC = ModelPredictive(
    step_size=one_minute * 15,
    nlp=NLP(
        model=model,
        N=32,
        objectives=[
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
    show_solution_plot=True,
    solution_plotter=mpc_plotter,
)

system.setup(start_time=0)
system.run(
    duration=one_day
)  # run the Simulation for one day without controller to settle the system

# Load initial data to consider for retraining
dh = load_DataHandler("pid_data")

for repetition in range(5):  # Start online learning loop
    ThermalZone_MPC.nlp.build(  # build nlp with (re-) trained models
        solver_options={"verbose": False, "ipopt.print_level": 0,'expand':True},
        predictors=[TAirRoom_predictor, Q_flowAhu_predictor],
    )

    online_data = system.run(  # run system with MPC for desired time
        controllers=[ThermalZone_MPC],
        duration=one_day * 1,
    )

    dh.add(online_data)  # add generated data to overall data for retraining
    mpc_plotter.plot(dh.containers[-1].df, show_plot=True)
    TAirRoom_TrainingData.clear()
    TAirRoom_TrainingData.add(dh)
    TAirRoom_TrainingData.shuffle()
    TAirRoom_TrainingData.split(0.7, 0.15, 0.15)
    TAirRoom_predictor.fit(  # Arguments of the fit function may vary depending on model type
        training_data=TAirRoom_TrainingData,
        # epochs=50,
        # batch_size=20,
        # verbose=1,
    )
    TAirRoom_predictor.test(training_data=TAirRoom_TrainingData)
    Q_flowAhu_TrainingData.clear()
    Q_flowAhu_TrainingData.add(dh)
    Q_flowAhu_TrainingData.shuffle()
    Q_flowAhu_TrainingData.split(0.8, 0.0, 0.2)
    Q_flowAhu_predictor.fit(training_data=Q_flowAhu_TrainingData)
    Q_flowAhu_predictor.test(training_data=Q_flowAhu_TrainingData)

dh.save("mpc_data", override=True) # save results
