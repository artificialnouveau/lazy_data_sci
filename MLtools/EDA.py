import pandas as pd
import numpy as np

from scipy.stats import pearsonr

import matplotlib.pyplot as plt 
import seaborn as sns

#Check for missing data
#=========================
def intitial_eda_checks(df):
    '''
    Purpose: general idea of the total and percentage of missing data in each column
    
    Input: Pandas dataframe

    Output: Prints string
    
    '''
    if df.isnull().sum().sum() > 0:
        mask_total = df.isnull().sum().sort_values(ascending=False) 
        total = mask_total[mask_total > 0]

        mask_percent = df.isnull().mean().sort_values(ascending=False) 
        percent = mask_percent[mask_percent > 0] 

        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
        print(f'Total and Percentage of NaN:\n {missing_data}')
    else: 
        print('No NaN found.')

def view_columns_w_many_nans(df, missing_percent_thres):
    '''
    Purpose:
    Checks which columns have over specified percentage of missing values

    Input:Takes df, missing percentage (0-1)

    Output:
    Returns columns as a list
    '''
    mask_percent = df.isnull().mean()
    series = mask_percent[mask_percent > missing_percent_thres]
    columns = series.index.to_list()
    print(columns) 
    return columns

def drop_columns_w_many_nans(df, missing_percent):
    '''
    Purpose:
    Takes df, missing percentage

    Input:
    Drops the columns whose missing value is bigger than missing percentage

    Output:
    Returns df
    
    '''
    series = view_columns_w_many_nans(df, missing_percent=missing_percent)
    list_of_cols = series.index.to_list()
    df.drop(columns=list_of_cols)
    print(list_of_cols)
    return df

#Plot distributions
#=============================
def histograms_numeric_columns(df):
    '''
    Takes df, numerical columns as list
    Returns a group of histagrams
    '''
    df_numeric = df.select_dtypes(include=[np.number])
    f = pd.melt(f_numeric) 
    g = sns.FacetGrid(f, col='variable',  col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, 'value')
    return g

  

#Correlation calculations
#==============================
def compute_corr_and_p(df1,df2):
  """
  Function calculates pearson correlation and statistical significance

  Input:
    df: pandas dataframes
    Note: Does not work with NaNs or non-numerical value
  
  Output:
    array of correlation r and statistical significance
  """
  corrs = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=np.float64)
  pvals = corrs.copy()
  
  for i, j in np.product(df1.columns, df2.columns):
    corrs.loc[i,j], pvals.loc[i,j] = pearsonr(df1[i], df2[j])
  
  return corrs, pvals

def heatmap_numeric_w_dependent_variable(df, dependent_variable, figx=8,figy=10):
    '''
    Takes df, a dependant variable as str
    Returns a heatmap of all independent variables' correlations with dependent variable 
    '''
    plt.figure(figsize=(figx, figy))
    g = sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable), 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1,
                    vmax=1) 
    return g


#Outlier detection
#=================================
def custom_zscore(df):
  df_zscore = df.select_dtypes(include=[np.number])
  df_zscore.columns = [x + "_zscore" for x in df_zscore.columns.tolist()]
  df_zscore=(df_zscore - df_zscore.mean())/df_zscore.std(axis=0, skipna = True)
  return df_zscore

def findoutliers(df,col,Q1lim,Q3lim):
  """
  Any data point which is less than Q1–1.5 IQR or Q3+1.5IQR are consider as outlier
  Qllim: low percentile range
  Q2lim: high percentile range
  """
  outliers=[]
  Q1=df[col].quantile(Q1lim)
  Q3=df[col].quantile(Q3lim)
  IQR=Q3-Q1
  low_limit=Q1-(1.5*IQR)
  up_limit=Q3+(1.5*IQR)
  for out1 in df[col]:
    if out1>up_limit or out1<low_limit:
      outliers.append(out1)
  return np.array(outliers)
  
