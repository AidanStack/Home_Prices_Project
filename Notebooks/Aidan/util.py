import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

def run_model(df_train, df_test, cols):
    from statsmodels.formula.api import ols
    
    X_train = df_train.loc[:,cols]
    X_test = df_test.loc[:,cols]
    
    formula = 'price ~ ' + ' + '.join(cols)
    
    model = ols(formula=formula, data=df_train).fit()
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    return model, train_preds, test_preds


def evaluate_model(df_train, df_test, cols):
    import statsmodels.api as sm

    model, train_preds, test_preds = run_model(df_train, df_test, ['sqft_living'])
    
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