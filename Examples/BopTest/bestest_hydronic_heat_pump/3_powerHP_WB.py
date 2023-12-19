from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *


power_hp_TrainingData_wb = TrainingData(
    inputs=Inputs(
        Input(source=u_hp, lag=1),
        Input(source=u_hp_logistic, lag=1),
        Input(source=t_amb, lag=1),
        Input(source=TAirRoom, lag=1),
    ),
    output=Output(power_hp),
    step_size=one_minute * 15,
)


power_hp_wb = WhiteBox(
    inputs=[u_hp, u_hp_logistic, t_amb, TAirRoom],
    output=power_hp,
    output_expression=(
        u_hp.source
        * 10000
        / ((TAirRoom.source + 15) / ((TAirRoom.source + 15) - t_amb.source) * 0.6)
        + 1110 * u_hp_logistic.source
    ),
    step_size=one_minute * 15,
)


pid_data = load_DataHandler("pid_data")

power_hp_TrainingData_wb.add(pid_data)
power_hp_TrainingData_wb.split(0., 0., 1)

power_hp_wb.test(training_data=power_hp_TrainingData_wb)
