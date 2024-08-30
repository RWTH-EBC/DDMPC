from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *


def run(training_data_name: str, name: str, training_data: TrainingData):

    pid_data = load_DataHandler(f'{training_data_name}')

    gpr = GaussianProcess(normalize=True)

    training_data, gpr = handle_training_data(
        training_data=training_data,
        data=pid_data,
        split={'trainShare': 0.8, 'validShare': 0.0, 'testShare': 0.2},
        trainer=gpr,
    )
    write_pkl(training_data, f'TrainingData_{name}_GPR', FileManager.data_dir())

    gpr.test(training_data=training_data)

    training_data.split(0.0, 0.0, 1.0)
    gpr.test(training_data=training_data)
    gpr.save(f'{name}_GPR', override=True)


def handle_training_data(training_data: TrainingData, data: DataHandler | DataContainer, split: dict,
                         trainer: GaussianProcess) -> [TrainingData, GaussianProcess]:
    """
    add data to TrainingData object, shuffle and split training_data then fit trainer

    :param training_data: TrainingData object
    :param data: data to add to the TrainingData object
    :param split: dict in the form {'trainShare': 0.8, 'validShare': 0, 'testShare': 0.2}
    :param trainer: GaussianProcess object
    """

    # add data to Training Data object
    # shuffle data and split into training, validation and testing sets
    training_data.add(data)
    training_data.shuffle()
    training_data.split(split['trainShare'], split['validShare'], split['testShare'])

    # train gpr
    trainer.fit(training_data=training_data)
    return training_data, trainer


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
