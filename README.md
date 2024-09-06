# Data Driven Model Predictive Control - DDMPC
Data driven Model Predictive Controllers use machine learning to predict future states of the system to optimize 
the control actions.

A brief explanation of the most important modules of the **ddmpc** package:

- **controller**: contains the core controller functionalities (PID, MPC, ...), cost types (Linear, AbsoluteLinear, Quadratic) and further classes to construct the NLP 
- **data_handling**: modules for storing, analysis, linearity detection and processing of training data
- **modeling**: modules for modeling, feature mapping and predicting, including OL. Different predictors available ML models (ANN, GPR, linReg) as well as physics based
- **systems**: everything to simulate building energy systems. Simulation interface can be accessed via abstract system class. Concrete implementation for FMUSystem and BopTest framework. 
- **utils**: utilities for formatting, plotting, logging, pickle handling etc as well as setting modes for controlled features.


# Step by step manual:

1. Create new folder for the system to be controlled
2. Create a **configuration** file, defining the system, the causalities and the variable mapping. Further configurations like modes for controlled features or plotting formats can be defined as well.
3. Define the training strategy and create training data by running the system with a baseline controller as displayed in **s2_generate_data** in Examples
4. Define the process models separately for each controlled feature (ANNs, GPRs, linReg, physics based) and fit the model with the generated training data (see e.g. **s3_TAirRoom_** in Examples)
5. Configure the MPC, define the NLP and run the system with or without online learning (see **s4_mpc** in Examples)

# Publications
To cite, please use:

P. Stoffel, M. Berktold, D. Müller, Real-Life Data-Driven Model Predictive Control for Building Energy Systems Comparing Different Machine Learning Models, Energy and Buildings (2024). https://doi.org/10.1016/j.enbuild.2024.113895

Further publications using the approach:

P. Stoffel, L. Maier, A. Kümpel, T. Schreiber, D. Müller, Evaluation of advanced control strategies for building energy systems, Energy and Buildings 280 (2023) 112709. https://doi.org/10.1016/j.enbuild.2022.112709              
    
  P. Stoffel, P. Henkel, M. Rätz, A. Kümpel, D. Müller, Safe operation of online learning data driven model predictive control of building energy systems, Energy and AI 14 (2023) 100296. https://doi.org/10.1016/j.egyai.2023.100296                            
  

