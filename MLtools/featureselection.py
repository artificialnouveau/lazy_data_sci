import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif_(X, thresh=5.0):
    '''
    Inspiration: https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-in-python
    
    Input: panda dataframe
    
    Output: Remaining columns after removing multicollinear features about threshold 5
    '''
    variables = list(range(X.shape[1]))
    dropped = True
    #Converts all columns to nan
    X=X.apply(pd.to_numeric, errors="coerce")
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] + '\'')
            del variables[maxloc]
            dropped = True

 

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]