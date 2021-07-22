

from MLtools import EDA

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


#dataframe EDA plots
#===================================
def numeric_eda(df, hue=None):
    """
    inspiration: https://gist.github.com/jiahao87/c97214065f996b76ab8fe4ca1964b2b5
    Given dataframe, generate EDA of numeric data
    Input: df - pandas dataframe
    Output: Pairplots
    
    """
    print("\nTo check: \nDistribution of numeric data")
    display(df.describe().T)
    columns = df.select_dtypes(include=np.number).columns
    figure = plt.figure(figsize=(20, 10))
    figure.add_subplot(1, len(columns), 1)
    for index, col in enumerate(columns):
        if index > 0:
            figure.add_subplot(1, len(columns), index + 1)
        sns.boxplot(y=col, data=df, boxprops={'facecolor': 'None'})
    figure.tight_layout()
    plt.show()
    
    if len(df.select_dtypes(include='category').columns) > 0:
        for col_num in df.select_dtypes(include=np.number).columns:
            for col in df.select_dtypes(include='category').columns:
                fig = sns.catplot(x=col, y=col_num, kind='violin', data=df, height=5, aspect=2)
                fig.set_xticklabels(rotation=90)
                plt.show()
    
    # Plot the pairwise joint distributions
    print("\nTo check pairwise joint distribution of numeric data")
    if hue==None:
        sns.pairplot(df.select_dtypes(include=np.number))
    else:
        sns.pairplot(df.select_dtypes(include=np.number).join(df[[hue]]), hue=hue)
    plt.show()
    
def categorical_eda(df, hue=None):
    """
    inspiration: https://gist.github.com/jiahao87/c97214065f996b76ab8fe4ca1964b2b5
    Given dataframe, generate EDA of categorical or non-numerical data
    Input: df - pandas dataframe
    Output: Catplots
    
    """
    
    print("\nTo check: \nUnique count of non-numeric data\n")
    print(df.select_dtypes(include=['object', 'category']).nunique())
    top5(df)
    # Plot count distribution of categorical data
    for col in df.select_dtypes(include='category').columns:
        fig = sns.catplot(x=col, kind="count", data=df, hue=hue)
        fig.set_xticklabels(rotation=90)
        plt.show()
        
        
#time series plots
#===================================
def time_series_plot(df):
    """
    inspiration: https://gist.github.com/jiahao87/c97214065f996b76ab8fe4ca1964b2b5
    Purpose: Given dataframe, generate times series plot of numeric data by daily, monthly and yearly frequency
    Input: df - pandas dataframe
    Output: times series plot
    """
    print("\nTo check time series of numeric data  by daily, monthly and yearly frequency")
    if len(df.select_dtypes(include='datetime64').columns)>0:
        for col in df.select_dtypes(include='datetime64').columns:
            for p in ['D', 'M', 'Y']:
                if p=='D':
                    print("Plotting daily data")
                elif p=='M':
                    print("Plotting monthly data")
                else:
                    print("Plotting yearly data")
                for col_num in df.select_dtypes(include=np.number).columns:
                    __ = df.copy()
                    __ = __.set_index(col)
                    __T = __.resample(p).sum()
                    ax = __T[[col_num]].plot()
                    ax.set_ylim(bottom=0)
                    ax.get_yaxis().set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                    plt.show()
                    
                    
#correlation plots
#===================================
def corrplot(df,size=10):
  """
  Function plots a graphical correlation matrix for each pair of columns in the dataframe.

  Input:
    df: pandas or spark DataFrame 
    size: vertical and horizontal size of the plot
  """
  
  if isinstance(df, pd.DataFrame):
    corr = df.corr()
  else:
    corr = df.toPandas().corr()
    
  fig, ax = plt.subplots(figsize=(size, size))
  ax.matshow(corr)
  plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
  plt.yticks(range(len(corr.columns)), corr.columns)
  plt.subplots_adjust(left=0.25)
  # Loop over data dimensions and create text annotations.
  for (i, j), z in np.ndenumerate(corr):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

  plt.show()

def corrplot_sigonly(df1,df2, siglevel=.05):
  """
  Function plots a heatmap with only statistical signifiance correlations

  Input:
    df: pandas dataframes
    statistical significance threshold: 0.05
  
  Output:
    seaborn correlation heatmap

  """

  corrs, pvals=EDA.compute_corr_and_p(df1, df2)
  mask = np.zeros_like(corrs, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True
  mask[pvals >= siglevel] = True
  masked_corr = corrs.loc[~np.all(mask, axis=1), ~np.all(mask, axis=0)]
  
  fig, ax = plt.subplots()
  

  sns.heatmap(corrs, mask=mask, center=0, square=True, linewidths=.5, annot=True, fmt=".1f",cbar_kws={'shrink': .5})
  
