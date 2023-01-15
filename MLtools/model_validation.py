# author: Ahnjili
# created: 01-11-2020
# last updated: 01-12-2022

# import libraries
import pandas as pd
import numpy as np
import time
from scipy import interp
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc, mean_absolute_error, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import GridSearchCV, accuracy_score, precision_score, recall_score


# ROC curves
# ===================================
def cv_ROC_plot(cvtype, num_cv, model_parameters, X_df, y_df):
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
    for train, test in cv.split(X_df, y_df):
        probas_ = classifier.fit(
            X_df.iloc[train], y_df.iloc[train]).predict_proba(X_df.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_df.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
             color='r', label='Chance', alpha=.8)

    mean_tpr = np.median(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (
        mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                     color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def nCrossVal_class_concensus(CV_outer, 
                              CV_inner, 
                              X, 
                              y, 
                              grpID, 
                              mod, 
                              paramSpace, 
                              score, 
                              scorer, 
                              percent_consensus=0.0, 
                              datscaler=StandardScaler(), 
                              print_inner=True):
    start = time.time()

    print(type(mod).__name__)

    # save cross validated results
    outer_results = list()
    outer_feats = list()
    outer_feats_coef = list()
    outer_best = list()
    outer_param = list()
    outer_testsubs = list()
    outer_results_acc = list()
    outer_results_pre = list()
    outer_results_rec = list()
    outer_results_f1 = list()
    outer_results_auc = list()
    results_df = pd.DataFrame()

    fold = 0
    if print_inner:
        print('>Performance on inner folds:')
    for train_ix, test_ix in CV_outer.split(X, y, groups=grpID):

        subs_train = grpID[train_ix]
        X_temp = X.copy()
        X_train, X_test = X_temp[train_ix, :], X_temp[test_ix]
        y_train, y_test = y[train_ix],  y[test_ix]

        pipeline = Pipeline([('scaler', datscaler), ('clf', mod)])
        search = GridSearchCV(pipeline, paramSpace,
                              cv=CV_inner,
                              scoring=score, refit=score)

        result = search.fit(X_train, y_train, groups=subs_train)
        best_model = result.best_estimator_

        # Refitting using the best model on all TRAINING data
        try:
            feats_inner = SelectFromModel(
                best_model.named_steps['clf'], prefit=True).get_support()
        except ValueError:
            feats_inner = np.ones(X.shape[1], dtype=bool)

        try:
            feats_inner_coef = best_model.named_steps['clf'].coef_[0]
        except:
            try:
                feats_inner_coef = best_model.named_steps['clf'].feature_importances_
            except:
                print('Could not get feature coefficients')

        yhat = best_model.predict(X_test)  # [:,feats_inner]
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        pre = precision_score(y_test, yhat)
        rec = recall_score(y_test, yhat)
        f1 = f1_score(y_test, yhat)
        auc = roc_auc_score(y_test, yhat)

        # store the result
        outer_results_acc.append(acc)
        outer_results_pre.append(pre)
        outer_results_rec.append(rec)
        outer_results_f1.append(f1)
        outer_results_auc.append(auc)
        outer_testsubs.append(grpID[test_ix])

        outer_feats.append(feats_inner)  # Save the features per outer-loop
        outer_feats_coef.append(feats_inner_coef)
        outer_best.append(best_model)  # Save the best model per outer-loop
        # Save the best parameters per outer-loop
        outer_param.append(result.best_params_)
        outer_score = scorer(y_test, yhat)
        outer_results.append(outer_score)

        if print_inner:
            print('>fold=%.0f,>acc=%.2f,>pre=%.2f,>rec=%.2f,>f1=%.2f,>auc=%.2f,cfg=%s' % (
                fold, acc, pre, rec, f1, auc, result.best_params_))
        fold += 1
    print('>Overall performance:')
    print('Mean Accuracy: %.1f (%.1f)' % (round(np.mean(outer_results_acc)*100, 2),
                                          round(np.std(outer_results_acc)*100, 1)))
    print('Mean Precision: %.1f (%.1f)' % (round(np.mean(outer_results_pre)*100, 2),
                                           round(np.std(outer_results_pre)*100, 2)))
    print('Mean Recall: %.1f (%.1f)' % (round(np.mean(outer_results_rec) * 100, 2),
                                        round(np.std(outer_results_rec)*100, 2)))
    print('Mean F1: %.1f (%.1f)' % (round(np.mean(outer_results_f1) * 100, 2),
                                    round(np.std(outer_results_f1)*100, 2)))
    print('Mean AUC: %.1f (%.1f)' % (round(np.mean(outer_results_auc) * 100, 2),
                                     round(np.std(outer_results_auc)*100, 2)))

    print('Minutes passed since start of this cell: ',
          round((time.time()-start)/60, 2))

    # get best model
    bst_i = np.argmax(outer_results)
    final_mod = SelectFromModel(
        outer_best[bst_i].named_steps['clf'], prefit=True)

    # Find concensus features
    all_feats = np.vstack(outer_feats)
    # 0.0 = all folds, range is from 0.0 to 1.0
    consensus_coef_i = np.sum(
        all_feats, 0) >= all_feats.shape[0]*percent_consensus
    X_new = X[:, consensus_coef_i]
    print("Number of consensus features selected by the outerfolds:",
          sum(consensus_coef_i))

    # Fit best model
    final_mod = outer_best[bst_i].fit(X_new, y)

    tempdf = pd.DataFrame({'Final Model': [final_mod],
                           'AccMean': [round(np.mean(outer_results_acc)*100, 2)],
                           'AccSTD': [round(np.std(outer_results_acc)*100, 2)],
                          'PreMean': [round(np.mean(outer_results_pre)*100, 2)],
                           'PreSTD': [round(np.std(outer_results_pre)*100, 2)],
                           'RecMean': [round(np.mean(outer_results_rec)*100, 2)],
                           'RecSTD': [round(np.std(outer_results_rec)*100, 2)],
                           'F1Mean': [round(np.mean(outer_results_f1)*100, 2)],
                           'F1STD': [round(np.std(outer_results_f1)*100, 2)],
                           'AUCMean': [round(np.mean(outer_results_auc)*100, 2)],
                           'AUCSTD': [round(np.std(outer_results_auc)*100, 2)], })
    results_df = results_df.append(tempdf, ignore_index=True)

    print('==============Complete==================')

    return outer_best, consensus_coef_i, final_mod, results_df, outer_testsubs, outer_feats, outer_feats_coef


def nCrossVal_reg_concensus(CV_outer, 
                            CV_inner,
                            X, 
                            y, 
                            grpID, 
                            mod, 
                            paramSpace, 
                            score, 
                            scorer, 
                            percent_consensus, 
                            datscaler=StandardScaler(), 
                            print_inner=True):
    start = time.time()

    print(type(mod).__name__)

    # save cross validated results
    outer_results = list()
    outer_feats = list()
    outer_feats_coef = list()
    outer_best = list()
    outer_param = list()
    outer_testsubs = list()

    # save cross validated results
    outer_results_r2 = list()
    outer_results_mae = list()

    results_df = pd.DataFrame()

    fold = 0
    if print_inner:
        print('>Performance on inner folds:')
    for train_ix, test_ix in CV_outer.split(X, y, groups=grpID):
        subs_train = grpID[train_ix]

        X_temp = X.copy()

        X_train, X_test = X_temp[train_ix, :], X_temp[test_ix]
        y_train, y_test = y[train_ix],  y[test_ix]

        pipeline = Pipeline([('scaler', datscaler), ('clf', mod)])

        search = GridSearchCV(pipeline, paramSpace,
                              cv=CV_inner,
                              scoring=score, refit=score)

        result = search.fit(X_train, y_train, groups=subs_train)
        best_model = result.best_estimator_

        # Refitting using the best model on all TRAINING data
        try:
            feats_inner = SelectFromModel(
                best_model.named_steps['clf'], prefit=True).get_support()
        except ValueError:
            feats_inner = np.ones(X.shape[1], dtype=bool)
        try:
            feats_inner_coef = best_model.named_steps['clf'].coef_
        except:
            feats_inner_coef = best_model.named_steps['clf'].feature_importances_

        # evaluate the model
        yhat = best_model.predict(X_test)
        r2s = r2_score(y_test, yhat)
        maes = mean_absolute_error(y_test, yhat)

        # store the result
        outer_results_r2.append(r2s)
        outer_results_mae.append(maes)
        outer_testsubs.append(grpID[test_ix])

        outer_feats.append(feats_inner)  # Save the features per outer-loop
        outer_feats_coef.append(feats_inner_coef)
        outer_best.append(best_model)  # Save the best model per outer-loop
        # Save the best parameters per outer-loop
        outer_param.append(result.best_params_)
        outer_score = scorer(y_test, yhat)
        outer_results.append(outer_score)

        if print_inner:
            print('>fold=%.0f,>r2=%.2f,>mae=%.2f,cfg=%s' %
                  (fold, r2s, maes, result.best_params_))
        fold += 1
    print('>Overall performance:')
    print('Mean R2: %.1f (%.1f)' % (
        round(np.mean(outer_results_r2), 1), round(np.std(outer_results_r2), 1)))
    print('Mean MAE: %.1f (%.1f)' % (
        round(np.mean(outer_results_mae), 1), round(np.std(outer_results_mae), 1)))
    print('Minutes passed since start of this cell: ',
          round((time.time()-start)/60, 2))

    # get best model
    bst_i = np.argmax(outer_results)
    final_mod = SelectFromModel(
        outer_best[bst_i].named_steps['clf'], prefit=True)

    # Find concensus features
    all_feats = np.vstack(outer_feats)
    # 0.0 = all folds, range is from 0.0 to 1.0
    consensus_coef_i = np.sum(
        all_feats, 0) >= all_feats.shape[0]*percent_consensus
    X_new = X[:, consensus_coef_i]
    print("Number of consensus features selected by the outerfolds:",
          sum(consensus_coef_i))

    # Fit best model
    final_mod = outer_best[bst_i].fit(X_new, y)

    tempdf = pd.DataFrame({'Final Model': [final_mod],
                           'R2Mean': [round(np.mean(outer_results_r2), 2)],
                           'R2STD': [round(np.std(outer_results_r2), 2)],
                          'MAEMean': [round(np.mean(outer_results_mae), 2)],
                           'MAESTD': [round(np.std(outer_results_mae), 2)]})
    results_df = results_df.append(tempdf, ignore_index=True)

    print('==============Complete==================')

    # return outer_best,coef_i,final_mod,results_df
    return outer_best, consensus_coef_i, final_mod, results_df, outer_testsubs, outer_feats, outer_feats_coef


def beautify_class_results(results_df):
    results_df['Accuracy'] = round(results_df['AccMean'], 2).astype(
        str)+'% (+/-'+round(results_df['AccSTD'], 2).astype(str)+'%)'
    results_df['Precision'] = round(results_df['PreMean'], 2).astype(
        str)+'% (+/-'+round(results_df['PreSTD'], 2).astype(str)+'%)'
    results_df['Recall'] = round(results_df['RecMean'], 2).astype(
        str)+'% (+/-'+round(results_df['RecSTD'], 2).astype(str)+'%)'
    results_df['F1'] = round(results_df['F1Mean'], 2).astype(
        str)+'% (+/-'+round(results_df['F1STD'], 2).astype(str)+'%)'
    results_df['AUC'] = round(results_df['AUCMean'], 2).astype(
        str)+'% (+/-'+round(results_df['AUCSTD'], 2).astype(str)+'%)'

    return results_df


def beautify_reg_results(results_df):
    results_df['R2'] = round(results_df['R2Mean'], 2).astype(
        str)+' (+/-'+round(results_df['R2STD'], 2).astype(str)+')'
    results_df['MAE'] = round(results_df['MAEMean'], 2).astype(
        str)+' (+/-'+round(results_df['MAESTD'], 2).astype(str)+')'
    return results_df


def outer_feats_each_fold_df(outer_feats_list, col_names, task_name):
    fold_feats_df = pd.DataFrame(
        columns=['task', 'fold', 'num_features', 'featindex', 'features'])
    i = 0
    for feats in outer_feats_list:
        fold_feats_df.loc[i, 'task'] = task_name
        fold_feats_df.loc[i, 'fold'] = i
        fold_feats_df.loc[i, 'featindex'] = feats
        fold_feats_df.loc[i, 'num_features'] = np.count_nonzero(feats)
        fold_feats_df.loc[i, 'features'] = np.array(col_names)[feats]
        i += 1

    feats_count_df = pd.DataFrame(pd.Series(
        [bg for bgs in fold_feats_df['features'] for bg in bgs]).value_counts()).reset_index()
    feats_count_df.columns = ['features', 'numfolds']
    feats_count_df['task'] = task_name

    no_fold_list = list(set(col_names) - set(feats_count_df['features']))
    if no_fold_list:
        for feat in no_fold_list:
            print('This feature did not appear in any folds:', feat)
            feats_count_df = feats_count_df.append(
                {'features': feat, 'task': task_name, 'numfolds': 0}, ignore_index=True)
        else:
            print('All features appeared at least in one fold')
    return fold_feats_df, feats_count_df


def outer_feats_coef_df(outer_feat_coef_list, col_names, task_name):
    feats_coef_df = pd.DataFrame(columns=['task', 'fold', 'features', 'coef'])
    i = 0
    for feats in outer_feat_coef_list:
        feat_coef_temp = pd.DataFrame(zip(col_names, np.transpose(
            outer_feat_coef_list[i])), columns=['features', 'coef'])
        feat_coef_temp['fold'] = i
        feat_coef_temp['task'] = task_name
        feats_coef_df = pd.concat([feats_coef_df, feat_coef_temp])
        i += 1

    return feats_coef_df
