from Examples.FMUs.ashrae.config import *
from keras.callbacks import EarlyStopping

"""
Train an ANN to learn the Temperature change using the generated training data
"""

# load data
pid_data = load_DataHandler("pid_data")

# Add data to training data class, shuffle and split
TAirRoom_TrainingData.add(pid_data)
TAirRoom_TrainingData.shuffle()
TAirRoom_TrainingData.split(trainShare=0.7, validShare=0.15, testShare=0.15)

# define tuner model for hyperparameter optimization
TAirRoom_TunerModel = TunerModel(
    TunerBatchNormalizing(),
    TunerDense(units=(8, 16)),  # Train networks with 4 or 8 neurons
    TunerDense(units=(8, 16), optional=True),
    name="TAirRoom_TunerModel",
)

trainer = NetworkTrainer()
trainer.build(
    n=5, keras_tuner=TAirRoom_TunerModel
)  # Train n different networks with the defined configuration

trainer.fit( # pass training data and training parameters
    training_data=TAirRoom_TrainingData,
    epochs=1000,
    batch_size=100,
    verbose=1,
    # callbacks=[EarlyStopping(patience=50, verbose=1)],
)
trainer.eval(training_data=TAirRoom_TrainingData, show_plot=True)

trainer.best.save("TAirRoom_ann", override=True) # save best network
