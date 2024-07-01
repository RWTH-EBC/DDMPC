from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *

training_data_name = 'pid_data'
pid_data = load_DataHandler(f'{training_data_name}')

TAirRoom_TrainingData.add(pid_data)
TAirRoom_TrainingData.split(1.0, 0.0, 0.0)
write_pkl(TAirRoom_TrainingData, 'TrainingData_TAir', FileManager.data_dir())

lin = LinearRegression()
lin.fit(training_data=TAirRoom_TrainingData)

TAirRoom_TrainingData.split(0.0, 0.0, 1.0)
lin.test(training_data=TAirRoom_TrainingData)

lin.print_coefficients(TAirRoom_TrainingData)
lin.save("TairRoom_linReg", override=True)
