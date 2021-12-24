###################
# Author: David Dov
# date: 23.12.21
###################

import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


 
class GridSearch():
    def __init__(self, Model: object, params:dict, num_folds:int, metric):
        self.Model = Model
        self.params = params
        self.metric = metric
        self.num_folds = num_folds


    def _get_params_df(self):
        param_df = pd.DataFrame(columns=self.params.keys())
        param_list = [self.params[key] for key in self.params.keys()]
        param_list = list(itertools.product(*param_list))
        for i, x in enumerate(param_list):
            param_df.loc[i] = x
        self.param_df = param_df


    def _get_results_trial(self, trial_params:dict, X_tr:np.array, y_tr:np.array, X_val:np.array, y_val:np.array) -> float:
        model = self.Model(**trial_params).fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:,1]
        return self.metric(y_val, y_pred)


    def fit(self, X, y):
        self._get_params_df()
        df = pd.DataFrame()
        kfold = KFold(n_splits=self.num_folds)
        for i,(tr_idx, val_idx) in enumerate(kfold.split(X)):
            df[f'result_{i}'] = self.param_df.apply(lambda row: self._get_results_trial(row, X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]), axis=1)
        self.df = pd.concat([self.param_df, df], axis=1)
    
    
    def get_best_params(self):
        self.df['result'] = self.df[[f'result_{i}' for i in range(self.num_folds)]].mean(axis=1)
        self.df = self.df.drop([f'result_{i}' for i in range(self.num_folds)], axis=1)
        return self.df[self.df.result == self.df.result.max()].iloc[0].drop('result').to_dict() # iloc is for the case of multiple parameters with the highest result
    


