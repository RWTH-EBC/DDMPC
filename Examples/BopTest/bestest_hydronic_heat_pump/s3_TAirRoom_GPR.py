from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *


def run(training_data_name: str, name: str, training_data: TrainingData) -> TrainingData:

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

    return training_data


if __name__ == '__main__':

    TrainingData = run(training_data_name='pid_data', name='TAirRoom', training_data=TAirRoom_TrainingData)
