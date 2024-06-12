try:
    from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
except ImportError:
    print("Examples module does not exist. Importing configuration directly from current folder.")
    from configuration import *

# PID controller for the HP
HP_PID = PID(
    y=TAirRoom,
    u=u_hp,
    step_size=one_minute * 15,
    Kp=0.1,
    Ti=60 * 60,
    # Td=0,
)

system.setup(
    start_time=0,
    warmup_period=one_week,
    active_control_layers={"oveHeaPumY_activate": 1},
)


# dc_identification = system.run(controllers=[HP_PID], duration=one_day * 7)
TAirRoom.mode = Random()
dc_random = system.run(controllers=[HP_PID], duration=one_day * 14)
dh = DataHandler(
    [
        # dc_identification,
        dc_random
    ]
)
# pid_plotter.plot(df=dc_identification.df, show_plot=True,save_plot=False)
pid_plotter.plot(df=dc_random.df, show_plot=True, save_plot=False)

dh.save(filename="pid_data", override=True)
