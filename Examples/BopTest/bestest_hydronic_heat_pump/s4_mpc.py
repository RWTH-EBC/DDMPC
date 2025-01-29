from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
from ddmpc.modeling.process_models.machine_learning import training
import numpy as np


def run(config: dict, predictors: list) -> [dict, dict]:

    TAirRoom.mode = TAirRoom_economic  # changes mode previously defined in configuration.py
    FileManager.experiment = f'{config['mpc_name']}'  # changes path data will be saved to from now on

    """ Initialize Model Predictive Controller """
    hhp_MPC = ModelPredictive(
        step_size=one_minute * 15,  # step size of controller
        nlp=NLP(                    # non linear problem
            model=model,
            N=config['N'],                    # prediction horizon
            objectives=[            # objective function
                Objective(feature=TAirRoom, cost=Quadratic(weight=20)),
                Objective(feature=costs_el, cost=Linear(weight=1)),
                Objective(feature=u_hp_change, cost=Quadratic(weight=0.1)),
            ],
            constraints=[
                Constraint(feature=u_hp, lb=0, ub=1),
            ],
        ),
        forecast_callback=system.get_forecast,
        solution_plotter=mpc_plotter,
        show_solution_plot=False,
        save_solution_plot=False,
        save_solution_data=True,
    )

    # store objectives and constraints in additional config that will be returned
    additional_config = {"objectives": [], "constraints": []}
    for objective in hhp_MPC.nlp.objectives:
        additional_config["objectives"].append(objective.get_config())
    for constraint in hhp_MPC.nlp.constraints:
        additional_config["constraints"].append(constraint.get_config())

    try:
        # set up the system
        # if no scenario is given, given start_time and warmup_period are used to initialize the system
        # otherwise the system is initialized based on the scenario-parameters (predefined in BOPTEST framework)
        system.setup(
            scenario={'electricity_price': config['price_scenario'],
                      'time_period': config['scenario']},
            active_control_layers={"oveHeaPumY_activate": 1},
        )

        # more solver options are set in config in __main__
        solver_options: dict = config['solver_options']
        solver_options.update({
            "verbose": False,
            "ipopt.print_level": 2,
            "expand": True,
        })

        df = None

        #  Online learning loop
        for repetition in range(14):  # 14 days: standard period in BOPTEST to ensure comparability
            # build nonlinear problem with trained models
            # default algorithm: ipopt
            hhp_MPC.nlp.build(
                solver_options=solver_options, predictors=predictors
            )

            # runs the system for the given duration using the given MPC controller
            # duration has to be dividable by step size of the system
            # returns data frame (only current and not past data frames) in a DataContainer
            # plots data and saves plot to disk (directory: /stored_data/plots/[mpc_name]/
            online_data = system.run(controllers=(hhp_MPC,), duration=one_day * 1)
            online_data.plot(plotter=mpc_plotter, save_plot=True, save_name=f'mpc_{repetition}.png')

            # online learning
            for n, predictor_config in enumerate(config['predictors']):
                if predictor_config['online_learning']['use_online_learning']:
                    predictors[n] = training.online_learning(
                        data=online_data,
                        predictor=predictors[n],
                        split=predictor_config['online_learning']['split'] if 'split' in predictor_config['online_learning'].keys() else None,
                        clear_old_data=predictor_config['online_learning']['clear_old_data'],
                        **predictor_config['online_learning']['training_arguments'],
                    )
                    if isinstance(predictors[n], NeuralNetwork):
                        predictors[n].update_casadi_model()

            # concat data frame of current repetition to data frame of previous iterations if existing
            if df is None:
                df = online_data.df
            else:
                df = pd.concat([df, online_data.df], axis=0)
        system.close()

        # save data frame with data from all repetitions to file data.csv (directory: /stored_data/[mpc_name]/ )
        df.to_csv(str(Path(FileManager.experiment_dir(), 'data.csv')))

        # obtain / calculate kpis from system (calculated from start_time, not including warm_up period)
        # put kpis in data frame and save this to file kpis.csv (directory: /stored_data/[mpc_name]/ )
        kpis = system.get_kpis()
    finally:
        system.stop()  # stop run if BOPTEST service is used even in case an error occurs

    # calculate percentage of successful runs
    success = 0
    count = 0
    for element in df['success']:
        if not np.isnan(element):     # don't take into account NaN elements at the beginning
            count += 1
            if element is True:
                success += 1
    kpis['successful_runs'] = success / count

    # calculate mean runtime of solver
    total_runtime = 0
    count = 0
    for element in df['runtime']:
        if not np.isnan(element):  # don't take into account NaN elements at the beginning
            count += 1
            total_runtime += element
    kpis['runtime_mean'] = total_runtime / count

    kpis_df = pd.DataFrame(data=kpis, index=[0])
    kpis_df.to_csv(str(Path(FileManager.experiment_dir(), 'kpis.csv')), index=False)

    return kpis, additional_config


def load_predictor(predictor_config: dict) -> LinearRegression | NeuralNetwork | WhiteBox | GaussianProcess:

    # load predictors from disc
    if predictor_config['type'] == 'linReg':
        predictor: LinearRegression = load_LinearRegression(
            filename=f'{predictor_config['name']}_{predictor_config['type']}')
    elif predictor_config['type'] == 'ANN':
        # load best NN trained before
        predictor: NeuralNetwork = load_NetworkTrainer(
            filename=f'{predictor_config['name']}_{predictor_config['type']}').best
    elif predictor_config['type'] == 'GPR':
        predictor: GaussianProcess = load_GaussianProcess(
            filename=f'{predictor_config['name']}_{predictor_config['type']}')
    elif predictor_config['type'] == 'WB':
        # define white box predictor
        if predictor_config['name'] == 'TAirRoom':
            predictor: WhiteBox = WhiteBox(
                inputs=[t_amb.source, TAirRoom.source, u_hp.source, rad_dir.source],
                output=TAirRoom_change,
                output_expression=(one_minute * 15 / 70476480) *
                                  (-15000 * (TAirRoom.source - t_amb.source) / 35
                                   + 24 * rad_dir.source + 15000 * u_hp.source),
                step_size=one_minute * 15
            )
        elif predictor_config['name'] == 'powerHP':
            predictor: WhiteBox = WhiteBox(
                inputs=[u_hp.source, t_amb.source, TAirRoom.source, u_hp_logistic.source],
                output=power_hp,
                output_expression=(u_hp.source * 10000 *
                                   ((TAirRoom.source + 15 - t_amb.source) / ((TAirRoom.source + 15) * 0.55))
                                   + 1110 * u_hp_logistic.source),
                step_size=one_minute * 15,
            )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return predictor


if __name__ == '__main__':

    config = {
        'mpc_name': 'test',
        'scenario': 'peak_heat_day',
        'price_scenario': 'dynamic',
        'N': 48,                                    # prediction horizon
        'solver_options': {                         # more solver options are set in run()
            "ipopt.max_iter": 1000,
        },
        'predictors': [
            {
                'name': 'TAirRoom',
                'type': 'ANN',  # choose prediction type (ANN, GPR, linReg, WB) for room air temperature
                'online_learning': {
                    'use_online_learning': False,  # set False if online learning should not be used
                    'clear_old_data': False,  # set False if in OL the predictor should only be trained with new data
                    # 'split': {'trainShare': 0.7, 'validShare': 0.15, 'testShare': 0.15}, # if split not given, default values will be used
                    'training_arguments': {  # only relevant if predictor is ANN
                        'learning_rate': 1E-4,  # set learning rate for OL
                        'epochs': 100,
                        'batch_size': 50,
                        'verbose': 1,
                    },
                }
            },
            {
                'name': 'powerHP',
                'type': 'ANN',  # choose prediction type (ANN, GPR, linReg, WB) for power of heat pump
                'online_learning': {
                    'use_online_learning': False,  # set False if online learning should not be used
                    'clear_old_data': False,  # set False if in OL the predictor should only be trained with new data
                    # 'split': {'trainShare': 0.7, 'validShare': 0.15, 'testShare': 0.15}, # if split not given, default values will be used
                    'training_arguments': {  # only relevant if predictor is ANN
                        'learning_rate': 1E-4,  # set learning rate for OL
                        'epochs': 100,
                        'batch_size': 50,
                        'verbose': 1,
                    },
                },
            },
        ],
    }

    predictors = list()
    for pred_config in config['predictors']:
        predictors.append(load_predictor(pred_config))

    _, _ = run(config, predictors)
