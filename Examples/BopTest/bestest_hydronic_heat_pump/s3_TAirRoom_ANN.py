from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
from keras.callbacks import EarlyStopping

"""
Train an ANN to learn the Temperature change using the generated training data
"""


def run(training_data_name: str, name: str, training_data: TrainingData) -> TrainingData:

    # load DataHandler from pickle file saved in 2_generate_data
    pid_data = load_DataHandler(f'{training_data_name}')

    # add training data to Training Data object instantiated in configuration
    # shuffle data and split into training, validation and testing sets
    # write data into pickle file (same directory as pid_data file: /stored_data/data/ )
    training_data.add(pid_data)
    training_data.shuffle()
    training_data.split(trainShare=0.8, validShare=0.1, testShare=0.1)
    write_pkl(training_data, f'TrainingData_{name}_ANN', FileManager.data_dir())

    # Create a sequential Tuner Model for hyperparameter tuning
    tuner = TunerModel(
        TunerBatchNormalizing(),            # layer to normalize inputs
        TunerDense(units=(4, 8, 16)),              # layer can either have 4, 8 or 16 neurons
        # TunerDense(units=(4, 8), optional=True),
        name=name
    )

    # create Trainer and build n random neural networks based on above created tuner with the defined configuration
    trainer = NetworkTrainer()
    trainer.build(n=1, keras_tuner=tuner)

    # train all neural networks build above by passing training data and training parameters
    # print the configuration of the best network
    # evaluate the trained neural networks (printing, saving and plotting evaluation by default False)
    trainer.fit(
        training_data=training_data,
        epochs=1000,
        batch_size=100,                 # number of test samples propagated through the network at once
        verbose=1,                      # defines how the progress of the training is shown in terminal window
        callbacks=[EarlyStopping(patience=100, verbose=1, restore_best_weights=True)]
    )
    trainer.best.sequential.summary()
    trainer.eval(training_data=training_data, show_plot=True)

    # Saves all neural networks to pickle (directory: /stored_data/predictors/ )
    trainer.save(filename=f'{name}_ANN', override=True)

    return training_data


if __name__ == '__main__':

    TrainingData = run(training_data_name='pid_data', name='TAirRoom', training_data=TAirRoom_TrainingData)
