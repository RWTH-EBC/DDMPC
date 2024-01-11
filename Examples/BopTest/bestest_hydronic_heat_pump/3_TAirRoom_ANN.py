from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
from keras.callbacks import EarlyStopping

pid_data = load_DataContainer("pid_data")

TAirRoom_TrainingData.add(pid_data)
TAirRoom_TrainingData.shuffle()
TAirRoom_TrainingData.split(0.7, 0.15, 0.15)

ANNTrainer = NetworkTrainer()

tuner = TunerModel(
    TunerBatchNormalizing(),
    TunerDense(units=(4, 8, 16)),
    # TunerDense(units=(4, 8), optional=True),
    name="TAirRoom",
)

trainer = NetworkTrainer()
trainer.build(n=1, keras_tuner=tuner)

trainer.fit(
    training_data=TAirRoom_TrainingData,
    epochs=1000,
    batch_size=100,
    verbose=1,
    # callbacks=[EarlyStopping(patience=50, verbose=1, restore_best_weights=True)]
)

trainer.eval(training_data=TAirRoom_TrainingData, show_plot=True)

trainer.save(filename="TAirRoom_ANN", override=True)
