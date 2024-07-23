from Examples.FMUs.ashrae.config import *
from keras.callbacks import EarlyStopping

"""
Train an ANN to learn the AHUs Power using the generated training data
"""

# load DataHandler from pickle file saved in 2_generate_data
training_data_name = 'pid_data'
pid_data = load_DataHandler(f'{training_data_name}')

# add training data to Training Data object instantiated in configuration
# shuffle data and split into training, validation and testing sets
# write data into pickle file (same directory as pid_data file: /stored_data/data/ )
Q_flowAhu_TrainingData.add(pid_data)
Q_flowAhu_TrainingData.shuffle()
Q_flowAhu_TrainingData.split(trainShare=0.8, validShare=0.1, testShare=0.1)
write_pkl(Q_flowAhu_TrainingData, 'TrainingData_Q_flowAhu', FileManager.data_dir())

# Create a sequential Tuner Model for hyperparameter tuning
tuner = TunerModel(
    TunerBatchNormalizing(),                            # layer to normalize inputs
    TunerDense(units=(4, 8), activations=("relu",)),           # layer can either have 4 or 8 neurons
    # TunerDense(units=(4, 8), activations=("sigmoid", "relu"), optional=True),
    name="Q_flowAhu",
)

# create Trainer and build n random neural networks based on above created tuner with the defined configuration
trainer = NetworkTrainer()
trainer.build(n=1, keras_tuner=tuner)

# train all neural networks build above by passing training data and training parameters
# print the configuration of the best network
# evaluate the trained neural networks (printing, saving and plotting evaluation by default False)
trainer.fit(
    training_data=Q_flowAhu_TrainingData,
    epochs=1000,
    batch_size=100,                     # number of test samples propagated through the network at once
    verbose=1,                          # defines how the progress of the training is shown in terminal window
    callbacks=[EarlyStopping(patience=100, verbose=1, restore_best_weights=True)]
)
trainer.best.sequential.summary()
trainer.eval(training_data=Q_flowAhu_TrainingData, show_plot=True)

# Saves all neural networks to pickle (directory: /stored_data/predictors/ )
trainer.save(filename="Q_flowAhu_ann", override=True)
