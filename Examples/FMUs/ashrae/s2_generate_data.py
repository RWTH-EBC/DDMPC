from Examples.FMUs.ashrae.config import *

""" 
This script is used to generate the necessary training data
"""

name = 'pid_data'

TAirRoom.mode = TAirRoom_random  # changes mode previously defined in config.py


# Example Function (not used)
def tab_func(inp):
    return inp * 0.005 + 2


# not used
TabsFunc = CtrlFunction(
    function=tab_func,
    fun_in=dry_bul,
    fun_out=Q_flowTabs,
    step_size=one_minute * 15
)

# PID controller for the TABS
# step size of controller must be equals or greater than the step size of the system
# and must be dividable by the step size of the system
Tabs_PID = PID(
    y=TAirRoom,
    u=Q_flowTabs,
    step_size=one_minute * 15,
    Kp=1,
    Ti=900,
    Td=0,
)

# PID controller for the AHU
# step size of controller must be equals or greater than the step size of the system
# and must be dividable by the step size of the system
AHU_PID = PID(
    y=TAirRoom,
    u=TsetAHU,
    step_size=one_minute * 15,
    Kp=1,
    Ti=600,
    Td=0,
)

# set up the system
# here: default fmu instance and simulation tolerance used
system.setup(start_time=0)

# simulate the system for the given duration using the given base controllers
# duration has to be dividable by step size of the system
# returns data frame (only current and not past data frames) in a DataContainer
dh = DataHandler(
    [
        system.run(controllers=(AHU_PID, Tabs_PID), duration=one_week),
    ]
)

# plots generated training data of PID in the way defined in config.py
pid_plotter.plot(df=dh.containers[0].df, show_plot=True, save_plot=True, save_name=f'{name}.png')

# saves DataHandler with training data to pickle (directory: /stored_data/data/ )
dh.save(name, override=True)
