# Data Driven Model Predictive Control - DDMPC
Data driven Model Predictive Controllers use machine learning to predict future states of the system to optimize 
the control actions.

A brief explanation of the most important modules of the **ddmpc** package:

- **systems**: everything to simulate building energy systems. (FMUSystem, BopTest, ...) 
- **controller**: contains the core controller functionalities. (PID, MPC, ...) 
- **modeling**: modules for modeling, feature mapping and data handling. Simulation interface can be accessed via abstract system class
- **utils**: utilities for formatting, plotting and pickle handling.

# Step by step manual:

1. Create new folder for the system to be controlled
2. Create a **configuration** file, defining the system, the causalities and the variable mapping
3. Define the training strategy as displayed in **2_generate_data**
4. Define the process models (ANNs, GPRs, PhysicsBased, ...) to be used in **3_train**
5. Configure the MPC and the online learning loop in **4_mpc**