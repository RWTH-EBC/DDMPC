from Examples.FMUs.ashrae.config import *
from keras.callbacks import EarlyStopping

"""
Train an ANN to learn the Temperature change using the generated training data
"""

# load data
training_data_name = 'pid_data'
pid_data = load_DataHandler(f'{training_data_name}')

# Add data to training data class, shuffle and split
TAirRoom_TrainingData.add(pid_data)
TAirRoom_TrainingData.shuffle()
TAirRoom_TrainingData.split(trainShare=0.8, validShare=0.1, testShare=0.1)
write_pkl(TAirRoom_TrainingData, 'TrainingData_TAir', FileManager.data_dir())

# define tuner model for hyperparameter optimization
TAirRoom_TunerModel = TunerModel(
    TunerBatchNormalizing(),
    TunerDense(units=(8, 16), activations=("relu",)),  # Train networks with 4 or 8 neurons
    # TunerDense(units=(8, 16), optional=True),
    name="TAirRoom_TunerModel",
)

trainer = NetworkTrainer()
trainer.build(
    n=1, keras_tuner=TAirRoom_TunerModel
)  # Train n different networks with the defined configuration

trainer.fit( # pass training data and training parameters
    training_data=TAirRoom_TrainingData,
    epochs=1000,
    batch_size=100,
    verbose=1,
    callbacks=[EarlyStopping(patience=100, verbose=1, restore_best_weights=True)],
)
trainer.best.sequential.summary()
trainer.eval(training_data=TAirRoom_TrainingData, show_plot=True)
trainer.save(filename="TAirRoom_ann", override=True)
