from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *


def run(training_data_name: str, name: str, training_data: TrainingData):

    TAirRoom_pred = WhiteBox(
        inputs=[t_amb.source, TAirRoom.source, u_hp.source, rad_dir.source],
        output=TAirRoom_change,
        output_expression=(one_minute * 15 / 70476480) *
                          (-15000 * (TAirRoom.source - t_amb.source) / 35
                           + 24 * rad_dir.source + 15000 * u_hp.source),
        step_size=one_minute*15
    )

    pid_data = load_DataHandler(f'{training_data_name}')
    training_data.add(pid_data)
    training_data.split(0., 0., 1)

    TAirRoom_pred.test(training_data=training_data)
    # write_pkl(TAirRoom_pred,f'{name}_WB')


if __name__ == '__main__':

    # Define the Inputs and Outputs of the process models using the TrainingData class
    # Define training data for supervised machine learning
    # Room air temperature is controlled variable
    training_data = TrainingData(
        inputs=Inputs(
            Input(source=t_amb, lag=1),
            Input(source=TAirRoom, lag=1),
            Input(source=u_hp, lag=1),
            Input(source=rad_dir, lag=1),
        ),
        output=Output(TAirRoom_change),
        step_size=one_minute * 15,
    )
    
    run(training_data_name='pid_data', name='TAirRoom', training_data=training_data)
