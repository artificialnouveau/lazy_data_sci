
from MLtools import EDA

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


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
  plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
  plt.yticks(range(len(corr.columns)), corr.columns);
  plt.subplots_adjust(left=0.25)
  # Loop over data dimensions and create text annotations.
  for (i, j), z in np.ndenumerate(corr):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

  display(fig)

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
  

  sns.heatmap(corrs, mask=mask, center=0, square=True, linewidths=.5, annot=True, fmt=".1f",
           cbar_kws={'shrink': .5})
  
  display(fig)
