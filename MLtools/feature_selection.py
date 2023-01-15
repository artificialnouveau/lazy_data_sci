# author: Ahnjili
# created: 01-11-2020
# last updated: 01-12-2022

# import libraries
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS


def calculate_vif_(X, thresh = 5.0):
    '''
    Inspiration: https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-in-python

    Input: 

    X: panda dataframe containing all of the independent variables
    thres: VIF threshold, anything higher than the threshold will be removed

    Output: Remaining columns after removing multicollinear features about threshold 5
    '''
    variables = list(range(X.shape[1]))
    dropped = True
    # Converts all columns to nan
    X = X.apply(pd.to_numeric, errors="coerce")
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


def backwards_elimination(X, y, threshold = 0.05):
    """
    Purpose:
    We check the performance of the model and then iteratively remove the worst performing features one by one 
    till the overall performance of the model comes in acceptable range.
    The performance metric used here to evaluate feature performance is pvalue. 
    If the pvalue is above 0.05 (or any other threshold) then we remove the feature, else we keep it.

    Input:
    X: Pandas dataframe (independent variables only)
    y: Pandas dataframe (dependent variables)
    threshold: pvalue numerical variables

    Output:
    Subsets the X dataframe with the selected variables

    """
    # https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
    cols = list(X.columns)
    pmax = 1
    while (len(cols) > 0):
        p = []
        X_1 = X[cols].astype(float)
        X_1 = add_constant(X_1)
        model = OLS(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax > threshold):
            cols.remove(feature_with_p_max)
        else:
            break
    backwards_feat = cols
    print("Backwards Elimination Complete. Features Remaining:")
    print(len(backwards_feat), 'features')
    print('\n'.join(map(str, backwards_feat)))
    return X[backwards_feat]
