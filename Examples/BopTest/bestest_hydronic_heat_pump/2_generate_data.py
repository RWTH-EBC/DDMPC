from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *

name = 'pid_data'

TAirRoom.mode = TAirRoom_random

# PID controller for the HP
HP_PID = PID(
    y=TAirRoom,
    u=u_hp,
    step_size=one_minute * 15,
    Kp=1.5,
    Ti=6500,
    Td=0,
)

system.setup(
    start_time=0,
    warmup_period=one_week,
    active_control_layers={"oveHeaPumY_activate": 1},
)

dc_random = system.run(controllers=[HP_PID], duration=one_day * 14)
dh = DataHandler(
    [
        dc_random
    ]
)

pid_plotter.plot(df=dc_random.df, show_plot=True, save_plot=True, save_name=f'{name}.png')

dh.save(name, override=True)
