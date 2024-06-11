try:
    from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
except ImportError:
    print("Examples module does not exist. Importing configuration directly from current folder.")
    from configuration import *

pid_data = load_DataHandler("pid_data")

TAirRoom_TrainingData.add(pid_data)
TAirRoom_TrainingData.shuffle()
TAirRoom_TrainingData.split(0.8, 0.0, 0.2)

gpr = GaussianProcess(normalize=True)
gpr.fit(training_data=TAirRoom_TrainingData)
gpr.test(training_data=TAirRoom_TrainingData)

TAirRoom_TrainingData.split(0.0, 0.0, 1.0)
gpr.test(training_data=TAirRoom_TrainingData)
gpr.save("TairRoom_GPR", override=True)
