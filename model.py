import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.preprocessing


# Modeling section
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.metrics import explained_variance_score
from scipy import stats 



import seaborn as sns
import os


import wrangle as pp


def baseline(y_train, y_validate):
    """
    This funtion will create a baseline prediction taking mean and median value.
    
    """
    # Convert y_train and y_validate into a DataFrame
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # Train Mean
    pred_mean_train = y_train.sale_price.mean() 
    y_train['sale_price_pred_mean'] = pred_mean_train
    
    # Validate Mean
    pred_mean_validate = y_validate.sale_price.mean() 
    y_validate['sale_price_pred_mean'] = pred_mean_validate
    
     # Train Median
    pred_median_train = y_train.sale_price.median() 
    y_train['sale_price_pred_median'] = pred_median_train
    
    # Validate Median
    pred_median_validate = y_validate.sale_price.median() 
    y_validate['sale_price_pred_median'] = pred_median_validate
    
    # RMSE of sale_price_pred_mean
    rmse_train = mean_squared_error(y_train.sale_price, y_train.sale_price_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.sale_price, y_validate.sale_price_pred_mean)**(1/2)
    
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    # RMSE of sale_price_pred_median
    rmse_train = mean_squared_error(y_train.sale_price, y_train.sale_price_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.sale_price, y_validate.sale_price_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    
def linear_regression(X_train, y_train, X_validate, y_validate):
    """
    Function to perform linear regression on our data.
    """
    # Converting y to DataFrames
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # Create the model object
    lm = LinearRegression(normalize=True)
    
    # Fit the model
    lm.fit(X_train, y_train.sale_price)
    
    # Predict train
    y_train['sale_price_pred_lm'] = lm.predict(X_train)

    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.sale_price, y_train.sale_price_pred_lm) ** (1/2)

    # predict validate
    y_validate['sale_price_pred_lm'] = lm.predict(X_validate)

    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.sale_price, y_validate.sale_price_pred_lm) ** (1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))   
    

def lasso_lars(X_train, y_train, X_validate, y_validate, alpha):
    """
    LassoLars regression for our data.
    """
    
    # Cinverting y to DataFrames
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # create the model object
    lars = LassoLars(alpha)

    # fit the model to our training data.
    lars.fit(X_train, y_train.sale_price)

    # predict train
    y_train['sale_price_pred_lars'] = lars.predict(X_train)

    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.sale_price, y_train.sale_price_pred_lars)**(1/2)

    # predict validate
    y_validate['sale_price_pred_lars'] = lars.predict(X_validate)

    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.sale_price, y_validate.sale_price_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    
    
def Tweedie_regressor(X_train, y_train, X_validate, y_validate, power, alpha):
    '''Tweedie regressor for our data
    power = 0: Normal Distribution
    power = 1: Poisson Distribution
    power = (1,2): Compound Distribution
    power = 2: Gamma Distribution
    power = 3: Inverse Gaussian Distribution'''
    
    # Creating DataFrames for y
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    

    # create the model object
    glm = TweedieRegressor(power=power, alpha=alpha)

    # fit the model to our training data.
    glm.fit(X_train, y_train.sale_price)

    # predict train
    y_train['sale_price_pred_glm'] = glm.predict(X_train)

    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.sale_price, y_train.sale_price_pred_glm)**(1/2)

    # predict validate
    y_validate['sale_price_pred_glm'] = glm.predict(X_validate)

    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.sale_price, y_validate.sale_price_pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=", power, " & alpha=", alpha, 
        "\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))

    
    
    
def polynomial_regression(X_train, y_train, X_validate, y_validate, degree):
    """
    Function to model the our data using a polynomial linear regression
    """
    
    # Converting y to DataFrame
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree= degree)


    

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    
    # X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.sale_price)

    # predict train
    y_train['sale_price_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.sale_price, y_train.sale_price_pred_lm2)**(1/2)

    # predict validate
    y_validate['sale_price_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.sale_price, y_validate.sale_price_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=", degree, "\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)
    
 


    
def polynomial_regression_test(X_train, y_train, X_validate, y_validate, X_test, y_test, degree):
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)


    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.sale_price)

    # predict test
    y_test['sale_price_pred_lm2'] = lm2.predict(X_test_degree2)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.sale_price, y_test.sale_price_pred_lm2)**(1/2)

    # predict validate
    y_validate['sale_price_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.sale_price, y_validate.sale_price_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\Test: ", round(rmse_test), 
      "\nValidation/Out-of-Sample: ", rmse_validate)