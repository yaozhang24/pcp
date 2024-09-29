# Posterior Conformal Prediction (PCP)

This repository contains the Python code to implement PCP and reproduce the experiments and figures in our article [Posterior Conformal Prediction](https://candes.su.domains/publications/).

### Prerequisites

* numpy
* random
* pandas
* scipy
* scikit-learn
* statistics
* statsmodels
* tqdm
* tensorflow
* keras
* conditionalconformal

### Installing

The development version of our code is available on github:
```bash
git clone https://github.com/yaozhang24/pcp.git
```
### Example
One simple way to start using PCP is to generate intervals with approximate conditional coverage in our [synthetic data experiment](https://github.com/yaozhang24/pcp/blob/main/Fig_04_05_06_setting_1.ipynb):

```python


import numpy as np
from utils import PCP, train_val_test_split, simulate_data, cross_val_residuals
from sklearn.ensemble import RandomForestRegressor

# Generate a synthetic dataset and split it into three folds
X, Y = simulate_data(num_samples=15000, setting=1)
X_train, X_val, X_test, Y_train, Y_val, Y_test, _ = train_val_test_split(X, Y, 1/3)

# Train the random forest model on the full training data
RF = RandomForestRegressor().fit(X_train, Y_train)

# Get predictions and residuals for the validation and test sets
predictions_val = RF.predict(X_val)
R_val = np.abs(Y_val - predictions_val)
predictions = RF.predict(X_test)
R_test = np.abs(Y_test - predictions)

# Cross-validation to generate a separate set of residuals for hyperparameter selection
RF_model = RandomForestRegressor()
X_train_cv, R_train_cv = cross_val_residuals(X_train, Y_train, model=RF_model)

# Run PCP
alpha = 0.1  # Level for PCP
PCP_model = PCP()
PCP_model.train(X_train_cv, R_train_cv) # Hyperparameter selection
pcp_quantiles = PCP_model.calibrate(X_val, R_val, X_test, R_test, alpha)[0] # Compute quantiles

# Compute intervals for all test samples
lower_bounds = predictions - np.array(pcp_quantiles)
upper_bounds = predictions + np.array(pcp_quantiles)

```
PCP can also be applied to achieve robust subgroup coverage, and level-adaptive coverage in classification. 
The implementation of PCP in these applications follows the same steps above.
We refer to our real-data experiments ([MEPS19](https://github.com/yaozhang24/pcp/blob/main/Fig_09_subgroup_gender.ipynb) and [HAM10000](https://github.com/yaozhang24/pcp/blob/main/Fig_10_logistic.ipynb)) for a demonstration.


### Reproduction

To reproduce the experiments and figures in our article, please download the following datasets and run the corresponding notebooks in our repository.


### Datasets

Communities and Crime ([link](https://archive.ics.uci.edu/dataset/183/communities+and+crime))

Communities and Crime Unnormalized ([link](https://archive.ics.uci.edu/dataset/211/communities+and+crime+unnormalized))

Online News Popularity ([link](https://archive.ics.uci.edu/dataset/332/online+news+popularity))

Superconductivty ([link](https://archive.ics.uci.edu/dataset/464/superconductivty+data))

Medical Expenditure Panel Survey (MEPS) 19 & 20 ([link](https://github.com/yromano/cqr/tree/master?tab=readme-ov-file))

HAM10000 image dataset ([link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000))


