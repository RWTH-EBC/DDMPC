from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *


def run(training_data_name: str, name: str, training_data: TrainingData):

    power_hp_wb = WhiteBox(
        inputs=[u_hp.source, u_hp_logistic.source, t_amb.source, TAirRoom.source],
        output=power_hp,
        output_expression=(u_hp.source * 10000 *
                           ((TAirRoom.source + 15 - t_amb.source) / ((TAirRoom.source + 15) * 0.55))
                           + (1110 + 500) * u_hp_logistic.source),
        step_size=one_minute * 15,
    )

    pid_data = load_DataHandler(f'{training_data_name}')
    training_data.add(pid_data)
    training_data.split(0., 0., 1)

    power_hp_wb.test(training_data=training_data)


if __name__ == '__main__':

    # Define the Inputs and Outputs of the process models using the TrainingData class
    # Define training data for supervised machine learning
    # power of heat pump is controlled variable
    training_data = TrainingData(
        inputs=Inputs(
            Input(source=u_hp, lag=1),
            Input(source=u_hp_logistic, lag=1),
            Input(source=t_amb, lag=1),
            Input(source=TAirRoom, lag=1),
        ),
        output=Output(power_hp),
        step_size=one_minute * 15,
    )

    run(training_data_name='pid_data', name='powerHP', training_data=training_data)
