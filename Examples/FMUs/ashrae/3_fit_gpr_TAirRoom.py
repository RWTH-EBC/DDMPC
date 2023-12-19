from Examples.FMUs.ashrae.config import *
from ddmpc.data_handling.reduction import NystroemReducer

"""
Train a GPR to learn the room temperature change
using the generated training data
"""

# load data
pid_data = load_DataHandler("pid_data")

# Add data to training data class, shuffle and split
TAirRoom_TrainingData.add(pid_data)
TAirRoom_TrainingData.shuffle()
TAirRoom_TrainingData.split(trainShare=0.8, validShare=0.0, testShare=0.2)

# Use Nyström Reduction to reduce data set to 1000 Inducing Points
TAirRoom_TrainingData.reduce(NystroemReducer(n_components=500))

# Train GPR
gpr = GaussianProcess.find_best_GPR(
    training_data=TAirRoom_TrainingData,
    normalize=True,
    iterations=1,
)

gpr.test(TAirRoom_TrainingData)

# Save GPR
gpr.save(filename="TAirRoom_GPR", override=True)
