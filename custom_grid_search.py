###################
# Author: David Dov
# date: 23.12.21
###################

import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class CustomGridSearch():
    def __init__(self, Model: object, params:dict, train_folds, metric):
        self.Model = Model
        self.params = params
        self.metric = metric
        self.train_folds = train_folds #int or list
        self.num_folds = train_folds if type(train_folds)==int else len(train_folds)
        self._get_params_df()


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


    def get_best_params(self):
        self.df['result'] = self.df[[f'result_{i}' for i in range(self.num_folds)]].mean(axis=1)
        self.df = self.df.drop([f'result_{i}' for i in range(self.num_folds)], axis=1)
        return self.df[self.df.result == self.df.result.max()].iloc[0].drop('result').to_dict() # iloc is for the case of multiple parameters with the highest result
    
    
    def fit(self, X, y):
        df = self.param_df.copy()
        df[[f'result_{i}' for i in range(self.num_folds)]] = None
        kfold = KFold(n_splits=self.num_folds).split(X) if type(self.train_folds) == int else self.train_folds
        
        for i in range(len(df)):
            for j,(tr_idx, val_idx) in enumerate(kfold):
                if df.loc[i, [f'result_{j}']].isnull().values:
                    df.loc[i, [f'result_{j}']] = self._get_results_trial(self.param_df.loc[i], X[tr_idx], y[tr_idx], X[val_idx], y[val_idx])
    
        self.df = df
        self.best_params_ = self.get_best_params()
    


