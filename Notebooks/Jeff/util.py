import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from statsmodels.formula.api import ols
import statsmodels.api as sm


def run_model(df_train, df_test, cols, log_price):
    
    X_train = df_train.loc[:,cols]
    X_test = df_test.loc[:,cols]
    
    if log_price == True:
        formula = 'price_log ~ ' + ' + '.join(cols)
    else:
        formula = 'price ~ ' + ' + '.join(cols)
    
    model = ols(formula=formula, data=df_train).fit()
    
    if log_price == True:
        train_preds = np.exp(model.predict(X_train))
        test_preds = np.exp(model.predict(X_test))
    else:
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
    return model, train_preds, test_preds


def evaluate_model(df_train, df_test, cols, log_price):

    model, train_preds, test_preds = run_model(df_train, df_test, cols, log_price)
    
    y_train = df_train['price']
    y_test = df_test['price']
    
    train_residuals = y_train - train_preds
    test_residuals = y_test - test_preds
    
    print(f"Train R2: {metrics.r2_score(y_train, train_preds):.3f}")
    print(f"Test R2: {metrics.r2_score(y_test, test_preds):.3f}")
    print("****")
    print(f"Train RMSE: {metrics.mean_squared_error(y_train, train_preds, squared=False):,.0f}")
    print(f"Test RMSE: {metrics.mean_squared_error(y_test, test_preds, squared=False):,.0f}")
    print("****")
    print(f"Train MAE: {metrics.mean_absolute_error(y_train, train_preds):,.0f}")
    print(f"Test MAE: {metrics.mean_absolute_error(y_test, test_preds):,.0f}\n")
    
    
    # Plot the residuals
    print("Residual scatter plot:")
    plt.scatter(train_preds, train_residuals, label='Train')
    plt.scatter(test_preds, test_residuals, label='Test')

    # Scatter plot of residuals
    plt.axhline(y=0, color = 'red')
    plt.xlabel('predictions')
    plt.ylabel('residuals')
    plt.legend()
    plt.show()
    print('\n')
    print("Residual qq plot")
    sm.qqplot(train_residuals, line = 'r');

# Filter for only multicollinearity above a certain threshold
def high_corr(df, threshold):
    mult_corr = df.corr().abs().stack().reset_index().sort_values(0, ascending=False)

    # zip the variable name columns (Which were only named level_0 and level_1 by default) in a new column named "pairs"
    mult_corr['pairs'] = list(zip(mult_corr.level_0, mult_corr.level_1))

    # set index to pairs
    mult_corr.set_index(['pairs'], inplace = True)

    #d rop level columns
    mult_corr.drop(columns=['level_1', 'level_0'], inplace = True)

    # rename correlation column as cc rather than 0
    mult_corr.columns = ['cc']

    # drop duplicates. This could be dangerous if you have variables perfectly correlated with variables other than themselves.
    # for the sake of exercise, kept it in.
    mult_corr.drop_duplicates(inplace=True)

    highly_corr = mult_corr[(mult_corr.cc>threshold)] #& (test.cc <1)]

    return highly_corr