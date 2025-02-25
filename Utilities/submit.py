# -*- coding: utf-8 -*-
"""get_data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1D0Am-s4ekBuO6p9QJ7njNYQ_-3CFMLvV
"""

import pandas as pd
import numpy as np
import os
import sys
from google.colab import drive
drive.mount('/content/drive')

def submit(model_opti, test_data, test_data_pre_processed, target_increment = 0):
  predictions = model_opti.predict(test_data) - target_increment
  predictions = pd.DataFrame(predictions)

  predictions['HOME_WINS'] = (predictions[0] == 1).astype(int)
  predictions['DRAW'] = (predictions[0] == 0).astype(int)
  predictions['AWAY_WINS'] = (predictions[0] == -1).astype(int)

  predictions.index = test_data_pre_processed.index
  submission = predictions.reset_index()
  submission = submission.drop(columns=[0]).copy()

  return submission
