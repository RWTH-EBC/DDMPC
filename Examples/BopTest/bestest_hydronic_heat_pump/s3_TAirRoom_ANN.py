from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
from keras.callbacks import EarlyStopping
from ddmpc.modeling.process_models.machine_learning.training import handle_training_data_and_fit


"""
Train an ANN to learn the Temperature change using the generated training data
"""
def run(training_data_name: str, name: str, training_data: TrainingData):

    # Create a sequential Tuner Model for hyperparameter tuning
    tuner = TunerModel(
        TunerBatchNormalizing(),  # layer to normalize inputs
        TunerDense(units=(32,), activations=('softplus',)),  # layer has 32 neurons
        # TunerDense(units=(4, 8), optional=True),
        name=name
    )

    # create Trainer and build n random neural networks based on above created tuner with the defined configuration
    trainer = NetworkTrainer()
    trainer.build(n=1, keras_tuner=tuner)

    # load DataHandler from pickle file saved in 2_generate_data
    pid_data = load_DataHandler(f'{training_data_name}')

    trainer = handle_training_data_and_fit(
        training_data=training_data,
        data=pid_data,
        split={'trainShare': 0.8, 'validShare': 0.1, 'testShare': 0.1},
        trainer_or_predictor=trainer,
        epochs=10000,
        batch_size=32,  # number of test samples propagated through the network at once
        verbose=1,  # defines how the progress of the training is shown in console
        callbacks=[EarlyStopping(
            patience=100,
            verbose=1,
            restore_best_weights=True
        )]
    )

    # write data into pickle file (same directory as pid_data file: /stored_data/data/ )
    write_pkl(trainer.best.training_data, f'TrainingData_{name}_ANN', FileManager.data_dir(), override=True)

    # print the configuration of the best trained network
    # evaluate the trained neural networks (printing, saving and plotting evaluation by default False)
    trainer.best.sequential.summary()
    trainer.eval(training_data=trainer.best.training_data, show_plot=True)

    # Saves all neural networks to pickle (directory: /stored_data/predictors/ )
    trainer.save(filename=f'{name}_ANN', override=True)


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
