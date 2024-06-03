from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *

pid_data = load_DataHandler("pid_data")

TAirRoom_TrainingData.add(pid_data)

lin = LinearRegression()

TAirRoom_TrainingData.split(1.0, 0.0, 0.0)
lin.fit(training_data=TAirRoom_TrainingData)
TAirRoom_TrainingData.split(0.0, 0.0, 1.0)
lin.test(training_data=TAirRoom_TrainingData)

lin.print_coefficients(TAirRoom_TrainingData)

lin.save("TairRoom_linReg", override=True)
