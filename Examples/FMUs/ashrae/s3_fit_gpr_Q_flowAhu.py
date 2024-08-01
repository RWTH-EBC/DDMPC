from Examples.FMUs.ashrae.config import *

"""
Train a GPR to learn the AHUs Power using the generated training data
"""

# load data
pid_data = load_DataHandler("pid_data")

# Add data to training data class, shuffle and split
Q_flowAhu_TrainingData.add(pid_data)
Q_flowAhu_TrainingData.shuffle()
Q_flowAhu_TrainingData.split(trainShare=0.8, validShare=0.0, testShare=0.2)

# Train GPR
gpr = GaussianProcess.find_best_GPR(
    training_data=Q_flowAhu_TrainingData,
    normalize=True,
    iterations=1,
)

gpr.test(Q_flowAhu_TrainingData,show_plot=True)

# Save GPR
gpr.save(filename="Q_flowAhu_GPR", override=True)
