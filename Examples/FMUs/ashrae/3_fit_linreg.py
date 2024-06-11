try:
    from Examples.FMUs.ashrae.config import *
except ImportError:
    print("Examples module does not exist. Importing configuration directly from current folder.")
    from config import *

"""
Train linear regression models to learn the AHUs Power 
and the room temperature change using the generated training data
"""

# load data
pid_data = load_DataHandler("pid_data")

# Add data to training data class
TAirRoom_TrainingData.add(pid_data)

# Fit and save regression model
lin = LinearRegression()
TAirRoom_TrainingData.split(trainShare=1.0, validShare=0.0, testShare=0.0)
lin.fit(training_data=TAirRoom_TrainingData)
lin.test(training_data=TAirRoom_TrainingData)
lin.save("TAirRoom_linreg", override=True)

# Add data to training data class
Q_flowAhu_TrainingData.add(pid_data)

# Fit and save regression model
lin = LinearRegression()
Q_flowAhu_TrainingData.split(trainShare=1.0, validShare=0.0, testShare=0.0)
lin.fit(training_data=Q_flowAhu_TrainingData)
lin.test(training_data=Q_flowAhu_TrainingData)
lin.save("Q_flowAhu_linreg", override=True)
