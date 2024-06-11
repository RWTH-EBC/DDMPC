try:
    from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
except ImportError:
    print("Examples module does not exist. Importing configuration directly from current folder.")
    from configuration import *

pid_data = load_DataHandler("pid_data")

power_hp_TrainingData.add(pid_data)
power_hp_TrainingData.shuffle()
power_hp_TrainingData.split(0.8, 0.0, 0.2)


gpr = GaussianProcess(scale=3000, normalize=True)
gpr.fit(power_hp_TrainingData)
gpr.test(power_hp_TrainingData)
gpr.save("power_hp_GPR_500_IP", override=True)
