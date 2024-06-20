from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
from keras.callbacks import EarlyStopping

training_data_name = 'pid_data'

pid_data = load_DataHandler(f'{training_data_name}')

TAirRoom_TrainingData.add(pid_data)
TAirRoom_TrainingData.shuffle()
TAirRoom_TrainingData.split(0.8, 0.1, 0.1)
write_pkl(TAirRoom_TrainingData, 'TrainingData_TAir', FileManager.data_dir())

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
    callbacks=[EarlyStopping(patience=100, verbose=1, restore_best_weights=True)]
)
trainer.best.sequential.summary()
trainer.eval(training_data=TAirRoom_TrainingData, show_plot=True)
trainer.save(filename="TAirRoom_ANN", override=True)
