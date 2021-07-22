import pandas as pd
import numpy as np

import logging as log
import numbers
from joblib import Parallel, delayed

from scipy import stats,interp

import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve,auc,matthews_corrcoef, explained_variance_score,mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold, ParameterGrid, ParameterSampler
from sklearn.model_selection import cross_validate,accuracy_score,matthews_corrcoef,precision_score,recall_score
from sklearn.feature_selection import RFECV
from sklearn.utils.multiclass import type_of_target


#ROC curves
#===================================
def cv_ROCplot(cvtype, num_cv,model_parameters, X_df,y_df):

    """
    Function: Run classifier with cross-validation and plot ROC curves
    Input: 
    cvtype: Specify CV procedure for example:  StratifiedKFold(n_splits=num_cv)
    num_cv: Number of cross vlidations
    model_parameters: model that you want to fit 
    """
    
    cv = cvtype
    classifier = model_parameters
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(15, 7))

    i = 0
    roc_table=pd.DataFrame()
    feat_table=pd.DataFrame()
    for train, test in cv.split(X_df, y_df):
        probas_ = classifier.fit(X_df.iloc[train], y_df.iloc[train]).predict_proba(X_df.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_df.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
        j = np.arange(len(tpr))
        
        roc_temp = pd.DataFrame({'cv':i,'fpr' : pd.Series(fpr, index=j),'1-fpr' : pd.Series(1-fpr, index=j),'tpr' : pd.Series(tpr, index = j), 'thresholds' : pd.Series(thresholds, index = j)})

    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

    mean_tpr = np.median(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def custom_GridSearch_nestedCV(X,y,model,param,outerk,innerk,randomseed=None,task='classify',refit_score=None):
    """
    Purpose: To do nested cross validation with gridsearch 
    Input:
    X: pandas dataframe that contains the feature/independent variables data
    y: panda dataframe that contains the output/dependent variables data
    model: classification or regression model that will be used
    param: hyperparameter grid for the gridsearch
    outer and innerk: Number of k folds for outer and inner dataset split
    randomseed: to ensure that the split is reproducible, I would recommend setting a randomseed for the k-fold splits
    task: what is the model objective? classify or regress?
    refit_score: how do you want to optimize the gridsearch? viva fl_score,accuracy_score,precision_score,matthew_corrcoef?
    Output:
    Best_Params: All best params from each inner loop cumulated in a dict
    Averaged validation scores based on the outer loop
    """
    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=outerk, shuffle=True, random_state=randomseed)
    # enumerate splits
    gridsearch_bestscore=list()
    gridsearch_bestparams=list()
    if task=='classify':
        acc_results= list()
        mcc_results= list()
        prec_results= list()
        sen_results= list()
        spec_results= list()
    elif task=='regress':
        var_results=list()
        mse_results=list()
        mae_results=list()
    kfold=1
    for train_index, test_index in cv_outer.split(X,y):
        print('Doing outer fold: ',kfold)
        kfold+=1
        # split data
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        # configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=innerk, shuffle=True, random_state=randomseed)
        #print("Doing Inner Cross Validation Grid Search Now")
        clf = GridSearchCV(model, param,cv=cv_inner,return_train_score=True,n_jobs=-1,refit=refit_score)
        # execute search
        result = clf.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        
        #save score
        gridsearch_bestscore.append(result.best_score_)
        gridsearch_bestparams.append(result.best_params_)
        
        if task=='classify':
            # evaluate the model
            acc_ = accuracy_score(y_test, yhat)
            mcc_ = matthews_corrcoef(y_test, yhat)
            prec_ = precision_score(y_test, yhat)
            sen_ = recall_score(y_test, yhat)
            spec_ = recall_score(y_test, yhat, pos_label=0)
            # store the result
            acc_results.append(acc_)
            mcc_results.append(mcc_)
            prec_results.append(prec_)
            sen_results.append(sen_)
            spec_results.append(spec_)
        
        elif task=='regress':
            var_=explained_variance_score(y_test, yhat)
            mse_=mean_squared_error(y_test, yhat)
            mae_=mean_absolute_error(y_test, yhat)
            var_results.append(var_results)
            mse_results.append(mse_results)
            mae_results.append(mae_results)

        print('k-fold=%.1f, est=%.3f, cfg=%s' % (kfold, result.best_score_, result.best_params_))
        
    #Get best hyperparameters
    bestscoreindex=gridsearch_bestscore.index(max(gridsearch_bestscore))
    print('Best params: ',gridsearch_bestparams[bestscoreindex])
    
    if task=='classify':
        print('Accuracy: %.2f%% +/-(%.2f)' % (np.mean(acc_results)*100, np.std(acc_results)*100))
        print('Matthews Correlation Coefficent: %.2f%% +/-(%.2f)'% (np.mean(mcc_results)*100, np.std(mcc_results)*100))
        print('Precision: %.2f%% +/-(%.2f)' % (np.mean(prec_results)*100, np.std(prec_results)*100))
        print('Recall: %.2f%% (+/-%.2f)' % (np.mean(sen_results)*100, np.std(sen_results)*100))
        print('Specificity: %.2f%% +/-(%.2f)' % (np.mean(spec_results)*100, np.std(spec_results)*100))
        
    elif task=='regress':
        print('Variance Explained: %.2f%% +/-(%.2f)' % (np.mean(var_results)*100, np.std(var_results)*100))
        print('Mean Squared Error: %.2f%% +/-(%.2f)'% (np.mean(mse_results)*100, np.std(mse_results)*100))
        print('Mean Absolute Error: %.2f%% +/-(%.2f)' % (np.mean(mae_results)*100, np.std(mae_results)*100))


