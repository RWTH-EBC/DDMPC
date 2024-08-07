from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *


def run(training_data_name: str, name: str, training_data: TrainingData):

    pid_data = load_DataHandler(f'{training_data_name}')

    training_data.add(pid_data)
    training_data.shuffle()
    training_data.split(0.8, 0.0, 0.2)
    write_pkl(training_data, f'TrainingData_f{name}_GPR', FileManager.data_dir())

    gpr = GaussianProcess(normalize=True)
    gpr.fit(training_data=training_data)
    gpr.test(training_data=training_data)

    training_data.split(0.0, 0.0, 1.0)
    gpr.test(training_data=training_data)
    gpr.save(f'{name}_GPR', override=True)


if __name__ == '__main__':

    # Define the Inputs and Outputs of the process models using the TrainingData class
    # Define training data for supervised machine learning
    # Room air temperature is controlled variable
    training_data = TrainingData(
        inputs=Inputs(
            Input(source=TAirRoom, lag=3),
            Input(source=t_amb, lag=2),
            Input(source=rad_dir, lag=1),
            Input(source=u_hp, lag=3),
        ),
        output=Output(TAirRoom_change),
        step_size=one_minute * 15,
    )

    run(training_data_name='pid_data', name='TAirRoom', training_data=training_data)
