import os
import json

import pandas as pd

runs = [os.path.join('run', f) for f in os.listdir('run')]

data = []
index = []
for run in runs:
    with open(os.path.join(run, 'metrics.log'), 'r') as f:
        res = f.read()
    for r in res.strip().split('\n'):
        js = json.loads(r)
        index.append((run, js['iteration']))
        data.append(js)
index = pd.MultiIndex.from_tuples(index, names = ['Run', 'Iteration'])
df = pd.DataFrame(data, index=index)
sorted_df = df.groupby(level='Run').apply(lambda x: x.sort_values(by='val_loss').head(1))
sorted_df = sorted_df.reset_index(level=0, drop=True)
sorted_df.to_csv('results.csv')