from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *


def run(training_data_name: str, name: str, training_data: TrainingData) -> TrainingData:
    pid_data = load_DataHandler(f'{training_data_name}')

    training_data.add(pid_data)
    training_data.shuffle()
    training_data.split(0.8, 0.0, 0.2)

    gpr = GaussianProcess(scale=3000, normalize=True)
    gpr.fit(training_data)
    gpr.test(training_data)
    gpr.save(f'{name}_GPR_500_IP', override=True)

    return training_data


if __name__ == '__main__':

    TrainingData = run(training_data_name='pid_data', name='powerHP', training_data=power_hp_TrainingData)
