# hydra
Experiments on learning correction terms for hydration free energy calculations.

__datasets/:__ scripts and original data for feature and label generation.

__dGhydr_DNN/:__ scripts for training and testing DNN absolute hydration free energy protocol, output models and figures.

__dGhydr_SVM/:__ scripts for training and testing SVM absolute hydration free energy protocol, output models and figures.

__ddGhydr_SVM/:__ scripts for training and testing SVM relative hydration free energy protocol.

__General workflow:__ 
1. Execute scripts for feature and label generation. 
2. Run compile script to separate into training and external testing sets. 
3. Train using desired model type scripts.
4. Conduct external testing using the respective model type testing scripts. Convergence, scatter, and testingn vs training/validation fingerprint similarity plots automatically created.
