from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
from keras.callbacks import EarlyStopping

"""
Train an ANN to learn the Temperature change using the generated training data
"""


def run(training_data_name: str, name: str, training_data: TrainingData):

    # Create a sequential Tuner Model for hyperparameter tuning
    tuner = TunerModel(
        TunerBatchNormalizing(),  # layer to normalize inputs
        TunerDense(units=(4, 8, 16)),  # layer can either have 4, 8 or 16 neurons
        # TunerDense(units=(4, 8), optional=True),
        name=name
    )

    # create Trainer and build n random neural networks based on above created tuner with the defined configuration
    trainer = NetworkTrainer()
    trainer.build(n=1, keras_tuner=tuner)

    # load DataHandler from pickle file saved in 2_generate_data
    pid_data = load_DataHandler(f'{training_data_name}')

    training_data, trainer = handle_training_data(
        training_data=training_data,
        data=pid_data,
        split={'trainShare': 0.8, 'validShare': 0.1, 'testShare': 0.1},
        trainer=trainer,
        epochs=1000,
        batch_size=100,  # number of test samples propagated through the network at once
        verbose=1,  # defines how the progress of the training is shown in terminal window
        callbacks=[EarlyStopping(patience=100, verbose=1, restore_best_weights=True)]
    )
    # write data into pickle file (same directory as pid_data file: /stored_data/data/ )
    write_pkl(training_data, f'TrainingData_{name}_ANN', FileManager.data_dir())

    # print the configuration of the best trained network
    # evaluate the trained neural networks (printing, saving and plotting evaluation by default False)
    trainer.best.sequential.summary()
    trainer.eval(training_data=training_data, show_plot=True)

    # Saves all neural networks to pickle (directory: /stored_data/predictors/ )
    trainer.save(filename=f'{name}_ANN', override=True)


def handle_training_data(training_data: TrainingData, data: DataHandler | DataContainer, split: dict,
                         trainer: NetworkTrainer | NeuralNetwork, **training_arguments) -> [TrainingData, NetworkTrainer | NeuralNetwork]:
    """
    add data to TrainingData object, shuffle and split training_data then fit trainer

    :param training_data: TrainingData object
    :param data: data to add to the TrainingData object
    :param split: dict in the form {'trainShare': 0.8, 'validShare': 0.1, 'testShare': 0.1}
    :param trainer: either NetworkTrainer or NeuralNetwork object
    :param training_arguments: further arguments to pass on to the training
    """

    # add data to Training Data object
    # shuffle data and split into training, validation and testing sets
    training_data.add(data)
    training_data.shuffle()
    training_data.split(split['trainShare'], split['validShare'], split['testShare'])

    # train all neural networks build above / neural network given to function
    # by passing training data and training parameters
    trainer.fit(
        training_data=training_data,
        **training_arguments,
    )
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
