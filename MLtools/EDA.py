import pandas as pd
import numpy as np

from scipy.stats import pearsonr

import matplotlib.pyplot as plt 
import seaborn as sns

#Check for missing data
#=========================
def initial_eda_checks(df):
    '''
    Purpose: general idea of the total and percentage of missing data in each column
    
    Input: Pandas dataframe

    Output: Prints string
    
    '''
    #first sums columns then sums total. This calculates how many nulls there are in total
    if df.isnull().sum(axis=1).sum() > 0:
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
    f = pd.melt(df_numeric) 
    g = sns.FacetGrid(f, col='variable',  col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, 'value')
    return g

  

#Correlation calculations
#==============================
def compute_corr_and_p(df,col1,col2):
    """
    Function calculates pearson correlation and statistical significance
    Note: Does not work with non-numerical value and it drops rows with NaN
    
    
    Input:
    df: pandas dataframes
    col1,col2: two columns of interest, these are string values 'Age','Severity'
    
  
    Output:
    array of correlation r and statistical significance
    """
    df_=df.dropna(subset=[col1,col2])
    pearson_coef, p_value = pearsonr(df_[col1], df_[col2]) #define the columns to perform calculations on
    del df_
    print("Pearson Correlation Coefficient: ", round(pearson_coef,3), "and a P-value of:", round(p_value,3)) # Results 
    return pearson_coef,p_value

def heatmap_numeric_w_dependent_variable(df, dependent_variable, figx=8,figy=10):
    '''
    Takes df, a dependant variable as str
    Returns a heatmap of all independent variables' correlations with dependent variable 
    '''
    plt.figure(figsize=(figx, figy))
    sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable), 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1,
                    vmax=1) 


#Outlier detection
#=================================
def create_zscoredf(df):
  """
  Purpose:
  Calculates zscore for entire datafarme

  Input:
  df: pandas dataframe

  Output:
  dataframe with converted z-scores
  """
  
  #only calculate the zscores for columns that are floats/integers
  df_zscore = df.select_dtypes(include=[np.number])
  
  #add _zscore suffix so users know which columns were converted to zscores
  df_zscore.columns = [x + "_zscore" for x in df_zscore.columns.tolist()]
  
  #calculate zscore
  df_zscore=(df_zscore - df_zscore.mean())/df_zscore.std(axis=0, skipna = True)
  df_zscore=df.select_dtypes(exclude=[np.number]).join(df_zscore)
  return df_zscore

def handle_outliers_zscore(df,thres=3,handle='select'):
  """
  Purpose:
  Select rows that do not have outliers

  Input:
  df: pandas dataframe
  threshold: zscore threshold
  handle: choose to 'select' or 'remove' rows that do not meet your zscore threshold

  Output:
  df with rows that have outliers
  """
  
  #only calculate the zscores for columns that are floats/integers
  df_zscore = df.select_dtypes(include=[np.number])
  
  if handle=='select':
    #select rows that have zscores below the threshold
    df_zscore=df_zscore[df_zscore.apply(lambda x: np.abs(x - x.mean()) / x.std() < thres).all(axis=1)]
  elif handle=='remove':
    #select rows that have zscores that are higher than the threshold
    df_zscore=df_zscore[df_zscore.apply(lambda x: np.abs(x - x.mean()) / x.std() >= thres).all(axis=1)]
    
  #join with non-numerical columns that were previously excluded
  df_zscore=df.select_dtypes(exclude=[np.number]).join(df_zscore)
  return df_zscore


def handle_outliers_quartile(df,Q1lim,Q3lim,handle='select'):
  """
  Purpose:
  Select rows that do or do not have outliers based on quartile range

  Input:
  df: pandas dataframe
  Q1lim: Q1 threshold
  Q3lim: Q3 threshold
  handle: choose to 'select' or 'remove' rows that do not meet your zscore threshold

  Output:
  df with rows that have outliers
  """
    
  df_quart = df.select_dtypes(include=[np.number])
  Q1=df_quart.quantile(Q1lim)
  Q3=df_quart.quantile(Q3lim)
  IQR=Q3-Q1
  low_limit=Q1-(1.5*IQR)
  up_limit=Q3+(1.5*IQR)
  if handle=='select':
    #select rows that have values that are below the threshold
    df_quart = df_quart[(df_quart>low_limit)&(df_quart<up_limit)]
  elif handle=='remove':
    #select rows that have values that are above the threshold
    df_quart = df_quart[(df_quart<=low_limit)|(df_quart>=up_limit)]
  df_quart=df.select_dtypes(exclude=[np.number]).join(df_quart)
  return df_quart

#PCA analysis
#=====================================
def pcaplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]

    plt.scatter(xs ,ys) #without scaling
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid('off')

def pcaplot_labels(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]

    plt.scatter(xs ,ys) #without scaling
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 20, coeff[i,1] * 25, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 20, coeff[i,1] * 25, labels[i], color = 'g', ha = 'center', va = 'center')

            plt.xlim(-15,15)
            plt.ylim(-15,15)
            plt.xlabel("PC{}".format(1))
            plt.ylabel("PC{}".format(2))
            plt.grid('off')
