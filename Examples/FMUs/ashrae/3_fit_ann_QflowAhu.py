from Examples.FMUs.ashrae.config import *
from keras.callbacks import EarlyStopping

"""
Train an ANN to learn the AHUs Power using the generated training data
"""

# load data
pid_data = load_DataHandler("pid_data")

# Add data to training data class, shuffle and split
Q_flowAhu_TrainingData.add(pid_data)
Q_flowAhu_TrainingData.shuffle()
Q_flowAhu_TrainingData.split(trainShare=0.8, validShare=0.1, testShare=0.1)

# define tuner model for hyperparameter optimization
tuner = TunerModel(  # Train networks with one or two layers and sigmoid or relu activation function
    TunerBatchNormalizing(),
    TunerDense(units=(4, 8), activations=("sigmoid", "relu")),
    TunerDense(units=(4, 8), activations=("sigmoid", "relu"), optional=True),
    name="Q_flowAhu",
)

trainer = NetworkTrainer()
trainer.build(
    n=4, keras_tuner=tuner
)  # Train n different networks with the defined configuration

trainer.fit(  # pass training data and training parameters
    training_data=Q_flowAhu_TrainingData,
    epochs=1000,
    batch_size=100,
    verbose=1,
    # callbacks=[EarlyStopping(patience=50, verbose=1, restore_best_weights=True)]
)
trainer.eval(training_data=Q_flowAhu_TrainingData, show_plot=True)

trainer.best.save("Q_flowAhu_ann", override=True)  # save best network
