from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *

pid_data = load_DataContainer("pid_data")

TAirRoom_TrainingData.add(pid_data)
TAirRoom_TrainingData.shuffle()
TAirRoom_TrainingData.split(0.8, 0.0, 0.2)
TAirRoom_TrainingData.reduce(NystroemReducer(n_components=500))

gpr = GaussianProcess(normalize=True)
gpr.fit(training_data=TAirRoom_TrainingData)
gpr.test(training_data=TAirRoom_TrainingData)

TAirRoom_TrainingData.split(0.0, 0.0, 1.0)
gpr.test(training_data=TAirRoom_TrainingData)
gpr.save("TairRoom_GPR", override=True)
