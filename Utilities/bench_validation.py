# -*- coding: utf-8 -*-
"""
# Benchmark Validation
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

model_opti_logistic_regression = LogisticRegression(C=0.0030026315789473683, max_iter=2000, multi_class='multinomial', penalty='l1', solver='saga')
model_opti_random_forest = RandomForestClassifier(random_state=42, max_depth= 10, max_features= 'sqrt', min_samples_leaf= 2, min_samples_split= 5, n_estimators= 200)

def get_bench_logistic_regression():
    return model_opti_logistic_regression
    
def get_bench_random_forest():
    return model_opti_random_forest

def test_bench_random_forest(train_data, train_target, cv = 5, model = model_opti_random_forest, verbose = True):
  scores = cross_val_score(model, train_data, train_target, cv=cv)
  if(verbose):
    print("Accuracy:", scores.mean())
    print("Std: ", scores.std())
  return scores.mean()
  
def test_bench(train_data, train_target, cv = 5, model = model_opti_logistic_regression, verbose = True):
  scores = cross_val_score(model, train_data, train_target, cv=cv)
  if(verbose):
    print("Accuracy:", scores.mean())
    print("Std: ", scores.std())
  return scores.mean()