# __author__ = 'ktc312'
#  -*- coding: utf-8 -*-
# coding: utf-8

import pandas as pd
import numpy as np

bcw_df = pd.read_csv('breast-cancer-wisconsin.data', header=None)
col_names = ['id', 'clump_thickness', 'cell_size', 'cell_shape', 'MA', 'SE_cell_size',
             'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'malignant']
bcw_df.columns = col_names

bcw_df = bcw_df.drop('id', axis=1)

bcw_df.replace('?', np.nan, inplace=True)
bcw_df['bare_nuclei'] = bcw_df['bare_nuclei'].astype('float64')
bcw_df.malignant = bcw_df.malignant.map({4: 1, 2: 0})

bcw_df.to_pickle('bcw_df.pkl')
