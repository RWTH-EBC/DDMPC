from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *

""" 
This script is used to generate the necessary training data
"""

name = 'pid_data'

TAirRoom.mode = TAirRoom_identification  # changes mode previously defined in configuration.py

# PID controller for the HP
# step size of controller must be equals or greater than the step size of the system
# and must be dividable by the step size of the system
HP_PID = PID(
    y=TAirRoom,                     # Controlled
    u=u_hp,                         # Control
    step_size=one_minute * 15,
    Kp=1.5,
    Ti=6500,
    Td=0,
)

# set up the system
# if no scenario is given, given start_time and warmup_period are used to initialize the system
# otherwise the system is initialized based on the scenario-parameters (predefined in BOPTEST framework)
system.setup(
    start_time=0,               # start time has to be dividable by step size of the System
    warmup_period=one_week,     # warm up period not included in calculation of kpis
    active_control_layers={"oveHeaPumY_activate": 1},  # possible model inputs can be found in BOPTEST documentation
)

# simulate the system for the given duration using the given base controller
# duration has to be dividable by step size of the system
# returns data frame (only current and not past data frames) in a DataContainer
dc_random = system.run(controllers=[HP_PID], duration=one_day * 14)
dh = DataHandler(
    [
        dc_random
    ]
)

# plots generated training data of PID in the way defined in configuration.py
pid_plotter.plot(df=dc_random.df, show_plot=True, save_plot=True, save_name=f'{name}.png')

# saves DataHandler with training data to pickle (directory: /stored_data/data/ )
dh.save(name, override=True)

system.close()
system.stop()
