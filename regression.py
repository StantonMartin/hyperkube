import numpy as np
import pandas as pd

import os
import sklearn
import sklearn.metrics as met
from sklearn.model_selection import KFold

import utils

### ML models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV


#
# Regression
# Describe phenotypes using provided hyperspectal signatures.
# Input:
#       train_X - hyperspectral signatures of training samples
#       train_Y - target phenotypes of training samples
#       test_X - hyperspectral signatures of testing samples
#       test_Y - target phenotypes of testing samples
# Output: R2 score
# Note: The code is using Partial Least Square Regression model. You may switch to any other model available on Scikit-Learn.
#
def regression(train_X, train_Y, test_X, test_Y):
    params = {"n_components": np.linspace(10, 100, dtype="int32")}
    search = RandomizedSearchCV(
        estimator=PLSRegression(),
        n_iter=30,
        cv=5,
        n_jobs=-1,
        param_distributions=params,
        verbose=1,
        error_score="raise",
    )

    search.fit(train_X, train_Y)
    model = search.best_estimator_

    print("Best Parameter")
    print(search.best_params_)

    model.fit(train_X, train_Y)
    y_pred = model.predict(test_X)

    print(met.r2_score(test_Y, y_pred))
