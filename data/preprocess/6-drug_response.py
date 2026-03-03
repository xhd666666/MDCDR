import os

import numpy as np
import pandas as pd
import pubchempy as pcp

# Save in the form of three columns
gdsc2 = pd.read_csv('GDSC2.csv', index_col=0, header=0)
response_long_2  = gdsc2.stack().reset_index()
response_long_2 .columns = ['cell', 'drug', 'res']
response_long_2  = response_long_2 .dropna(subset=['res'])
response_long_2 .to_csv('response.csv', index=False, encoding='utf-8')

gdsc = pd.read_csv('GDSC.csv', index_col=0, header=0)
gdsc = gdsc.replace({'S': 1, 'R': 0})
response_long = gdsc.stack().reset_index()
response_long.columns = ['cell', 'drug', 'res']
response_long = response_long.dropna(subset=['res'])
response_long.to_csv('response_cls.csv', index=False, encoding='utf-8')

tcga = pd.read_csv('TCGA_filter.csv', header=0, index_col=0)
tcga = tcga.replace({'Complete Response': 1, 'Clinical Progressive Disease': 0,
                     'Partial Response': 1, 'Stable Disease': 0})
response_long = tcga.stack().reset_index()
response_long.columns = ['cell', 'drug', 'res']
response_long = response_long.dropna(subset=['res'])
response_long.to_csv('response_tcga.csv', index=False, encoding='utf-8')
