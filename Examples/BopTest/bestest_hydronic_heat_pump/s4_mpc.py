from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
import training as training


def run(config, t_air_room_pred, power_hp_pred) -> [dict, dict]:

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
    for repetition in range(14):  # for 14 days (standard period in BOPTEST to ensure comparability)
        # build nonlinear problem with trained models
        # default algorithm: ipopt
        hhp_MPC.nlp.build(
            solver_options=solver_options, predictors=[t_air_room_pred, power_hp_pred]
        )

        # runs the system for the given duration using the given MPC controller
        # duration has to be dividable by step size of the system
        # returns data frame (only current and not past data frames) in a DataContainer
        # plots data and saves plot to disk (directory: /stored_data/plots/[mpc_name]/
        online_data = system.run(controllers=(hhp_MPC,), duration=one_day * 1)
        online_data.plot(plotter=mpc_plotter, save_plot=True, save_name=f'mpc_{repetition}.png')

        # online learning room temperature
        if config['t_online_learning']['use_online_learning']:
            training.online_learning(
                data=online_data,
                predictor=t_air_room_pred,
                split=config['t_online_learning']['split'] if 'split' in config['t_online_learning'].keys() else None,
                **config['t_online_learning']['training_arguments'],
            )

        # online learning for power of heat pump
        if config['p_online_learning']['use_online_learning']:
            training.online_learning(
                data=online_data,
                predictor=power_hp_pred,
                split=config['p_online_learning']['split'] if 'split' in config['p_online_learning'].keys() else None,
                **config['p_online_learning']['training_arguments'],
            )

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
    kpis_df = pd.DataFrame(data=kpis, index=[0])
    kpis_df.to_csv(str(Path(FileManager.experiment_dir(), 'kpis.csv')), index=False)

    return kpis, additional_config

def load_predictor_t_air_room(config: dict) -> LinearRegression | NeuralNetwork | WhiteBox | GaussianProcess:

    # regression for TAirRoom, load predictors from disc
    if config['TAirRoom_pred_type'] == 'linReg':
        t_air_room_pred: LinearRegression = load_LinearRegression(filename=config['TAirRoom_pred_name'])
    elif config['TAirRoom_pred_type'] == 'ANN':
        # load best NN trained before
        t_air_room_pred: NeuralNetwork = load_NetworkTrainer(filename=config['TAirRoom_pred_name']).best
    elif config['TAirRoom_pred_type'] == 'GPR':
        t_air_room_pred: GaussianProcess = load_GaussianProcess(filename=config['TAirRoom_pred_name'])
    elif config['TAirRoom_pred_type'] == 'WB':
        # define white box predictor
        t_air_room_pred: WhiteBox = WhiteBox(
            inputs=[t_amb.source, TAirRoom.source, u_hp.source, rad_dir.source],
            output=TAirRoom_change,
            output_expression=(one_minute * 15 / 70476480) *
                              (-15000 * (TAirRoom.source - t_amb.source) / 35
                               + 24 * rad_dir.source + 15000 * u_hp.source),
            step_size=one_minute * 15
        )
    else:
        raise NotImplementedError()

    return t_air_room_pred


def load_predictor_power_hp(config: dict) -> LinearRegression | NeuralNetwork | WhiteBox | GaussianProcess:

    # regression for power_hp, load predictors from disc
    if config['power_hp_pred_type'] == 'linReg':
        power_hp_pred: LinearRegression = load_LinearRegression(filename=config['power_hp_pred_name'])
    elif config['power_hp_pred_type'] == 'ANN':
        # load best NN trained before
        power_hp_pred: NeuralNetwork = load_NetworkTrainer(filename=config['power_hp_pred_name']).best
    elif config['power_hp_pred_type'] == 'GPR':
        power_hp_pred: GaussianProcess = load_GaussianProcess(filename=config['power_hp_pred_name'])
    elif config['power_hp_pred_type'] == 'WB':
        # define white box predictor
        power_hp_pred: WhiteBox = WhiteBox(
            inputs=[u_hp.source, t_amb.source, TAirRoom.source, u_hp_logistic.source],
            output=power_hp,
            output_expression=(u_hp.source * 10000 *
                               ((TAirRoom.source + 15 - t_amb.source) / ((TAirRoom.source + 15) * 0.55))
                               + 1110 * u_hp_logistic.source),
            step_size=one_minute * 15,
        )
    else:
        raise NotImplementedError()

    return power_hp_pred


if __name__ == '__main__':

    config = {
        'mpc_name': 'test',
        'scenario': 'peak_heat_day',
        'price_scenario': 'dynamic',
        'TAirRoom_pred_type': 'linReg',             # choose prediction type (ANN, GPR, linReg, WB) for room air temperature
        'TAirRoom_pred_name': 'TAirRoom_linReg',    # name of predictor file saved on disc (.pkl)
        'power_hp_pred_type': 'linReg',             # choose prediction type (ANN, GPR, linReg, WB) for power of heat pump
        'power_hp_pred_name': 'powerHP_linReg',     # name of predictor file saved on disc (.pkl)
        'N': 48,                                    # prediction horizon
        'solver_options': {  # more solver options are set in run()
            "ipopt.max_iter": 1000,
        },
        't_online_learning': {                      # online learning for room air temperature
            'use_online_learning': True,            # set False if online learning should not be used
            # 'split': {'trainShare': 0.7, 'validShare': 0.15, 'testShare': 0.15}, # if split not given, default values will be used
            'training_arguments': {                 # only relevant if predictor is ANN
                'epochs': 100,
                'batch_size': 50,
                'verbose': 1,
            },
        },
        'p_online_learning': {                      # online learning for power of heat pump
            'use_online_learning': True,            # set False if online learning should not be used
            # 'split': {'trainShare': 0.7, 'validShare': 0.15, 'testShare': 0.15}, # if split not given, default values will be used
            'training_arguments': {                 # only relevant if predictor is ANN
                'epochs': 100,
                'batch_size': 50,
                'verbose': 1,
            },
        },
    }

    t_pred = load_predictor_t_air_room(config)
    p_pred = load_predictor_power_hp(config)

    _, _ = run(config, t_pred, p_pred)
