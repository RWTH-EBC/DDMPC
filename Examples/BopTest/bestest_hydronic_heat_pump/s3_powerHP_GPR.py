from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *


def run(training_data_name: str, name: str, training_data: TrainingData):
    pid_data = load_DataHandler(f'{training_data_name}')

    training_data.add(pid_data)
    training_data.shuffle()
    training_data.split(0.8, 0.0, 0.2)

    gpr = GaussianProcess(scale=3000, normalize=True)
    gpr.fit(training_data)
    gpr.test(training_data)
    gpr.save(f'{name}_GPR_500_IP', override=True)


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
