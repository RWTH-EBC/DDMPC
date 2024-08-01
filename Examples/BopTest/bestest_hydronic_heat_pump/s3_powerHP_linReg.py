from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *


def run(training_data_name: str, name: str, training_data: TrainingData) -> TrainingData:

    pid_data = load_DataHandler(f'{training_data_name}')

    training_data.add(pid_data)
    training_data.split(1.0, 0.0, 0.0)
    write_pkl(training_data, f'TrainingData_{name}_linReg', FileManager.data_dir())

    lin = LinearRegression()
    lin.fit(training_data=training_data)

    # TAirRoom_TrainingData.split(0.0, 0.0, 1.0)
    lin.test(training_data=training_data)

    lin.print_coefficients(training_data)
    lin.save(f'{name}_linReg', override=True)

    return training_data


if __name__ == '__main__':

    TrainingData = run(training_data_name='pid_data', name='powerHP', training_data=power_hp_TrainingData)
