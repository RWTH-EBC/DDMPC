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

# Publications
To cite, please use:

P. Stoffel, M. Berktold, D. Müller, Real-Life Data-Driven Model Predictive Control for Building Energy Systems Comparing Different Machine Learning Models, Energy and Buildings (2024). https://doi.org/10.1016/j.enbuild.2024.113895

Further publications using the approach:

P. Stoffel, L. Maier, A. Kümpel, T. Schreiber, D. Müller, Evaluation of advanced control strategies for building energy systems, Energy and Buildings 280 (2023) 112709. https://doi.org/10.1016/j.enbuild.2022.112709              
    
  P. Stoffel, P. Henkel, M. Rätz, A. Kümpel, D. Müller, Safe operation of online learning data driven model predictive control of building energy systems, Energy and AI 14 (2023) 100296. https://doi.org/10.1016/j.egyai.2023.100296                            
  

