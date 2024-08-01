from Examples.FMUs.ashrae.config import *

"""
Train linear regression models to learn the AHUs Power 
and the room temperature change using the generated training data
"""

# load data
training_data_name = 'pid_data'
pid_data = load_DataHandler(f'{training_data_name}')

# Add data to training data class
TAirRoom_TrainingData.add(pid_data)
TAirRoom_TrainingData.split(trainShare=1.0, validShare=0.0, testShare=0.0)
write_pkl(TAirRoom_TrainingData, 'TrainingData_TAir', FileManager.data_dir())

# Fit and save regression model
lin = LinearRegression()
lin.fit(training_data=TAirRoom_TrainingData)
lin.test(training_data=TAirRoom_TrainingData)
lin.save("TAirRoom_linreg", override=True)

# Add data to training data class
Q_flowAhu_TrainingData.add(pid_data)
Q_flowAhu_TrainingData.split(trainShare=1.0, validShare=0.0, testShare=0.0)
write_pkl(Q_flowAhu_TrainingData, 'TrainingData_Q_flowAhu', FileManager.data_dir())

# Fit and save regression model
lin = LinearRegression()
lin.fit(training_data=Q_flowAhu_TrainingData)
lin.test(training_data=Q_flowAhu_TrainingData)
lin.save("Q_flowAhu_linreg", override=True)
