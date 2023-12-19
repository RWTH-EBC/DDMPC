from Examples.FMUs.ashrae.config import *

""" 
This script is used to generate the necessary training data
"""

TAirRoom.mode = Random( # Set the Air temperature mode to random for identification
    day_start=8,
    day_end=16,
    day_lb=273.15 + 19,
    night_lb=273.15 + 16,
    day_ub=273.15 + 21,
    night_ub=273.15 + 24,
    interval=60 * 60 * 4,   # change set point after interval
    time_offset=0,
)

# PID controller for the TABS
Tabs_PID = PID(
    y=TAirRoom,
    u=Q_flowTabs,
    step_size=60 * 15,
    Kp=1,
    Ti=900,
    Td=0,
)

# PID controller for the AHU
AHU_PID = PID(
    y=TAirRoom,
    u=TsetAHU,
    step_size=60 * 15,
    Kp=1,
    Ti=600,
    Td=0,
)

# Simulate system for given time using the defined base controllers
system.setup(start_time=one_week * 0)
dh = DataHandler(
    [
        system.run(controllers=(AHU_PID, Tabs_PID), duration=one_week),
    ]
)

# Plot Training data
pid_plotter.plot(df=dh.containers[0].df, show_plot=True, save_plot=False)

# Save training data
dh.save("pid_data", override=True)
