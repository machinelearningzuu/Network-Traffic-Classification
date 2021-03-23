import re
from pathlib import Path
import pandas as pd

import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
from variables import*

current_dir = os.getcwd()
csv_folder = 'data'
csv_dir = os.path.join(current_dir, csv_folder)

def prepare_values(df):
    df_values = df[inout_columns].values
    return df_values

source_dir = Path(csv_dir)
in_data = []
for file in source_dir.glob('**/*.in.csv'):
    activity = {'activity': file.stem.split('.')[0]}
    df = pd.read_csv(file)
    cols = ['in-frame.time_delta_displayed', 'in-frame.len', 'in-data.len']
    values = prepare_values(df)
    for value in values:
        file_data ={**activity, **dict(zip(cols,values))}
        in_data.append(file_data)

out_data =[]
for file in source_dir.glob('**/*.out.csv'):
    activity = {'activity': file.stem.split('.')[0]}
    df = pd.read_csv(file)
    cols = ['out-frame.time_delta_displayed', 'out-frame.len', 'out-data.len']
    values = prepare_values(df)
    for value in values:
        file_data ={**activity, **dict(zip(cols,values))}
        out_data.append(file_data)

in_df = pd.DataFrame(in_data)
out_df = pd.DataFrame(out_data)
all_df = in_df.join(out_df.set_index('activity'), on='activity', how='outer')
all_df.dropna(subset=all_df.columns.tolist()[1:], how='all', inplace=True)
all_df.fillna(0, inplace=True)
all_df['activity'] = all_df['activity'].apply(lambda x:re.sub(r'^([a-zA-Z]+).*', r'\1',x))
all_df.to_csv('Fake_stat.csv', index=False)