from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
from s3_TAirRoom_linReg import handle_training_data

def run(training_data_name: str, name: str, training_data: TrainingData):

    pid_data = load_DataHandler(f'{training_data_name}')

    lin = LinearRegression()
    training_data, lin = handle_training_data(
        training_data=training_data,
        data=pid_data,
        split={'trainShare': 1.0, 'validShare': 0, 'testShare': 0},
        trainer=lin,
    )
    write_pkl(training_data, f'TrainingData_{name}_linReg', FileManager.data_dir())

    training_data.split(0.0, 0.0, 1.0)
    lin.test(training_data=training_data)

    lin.print_coefficients(training_data)
    lin.save(f'{name}_linReg', override=True)


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
