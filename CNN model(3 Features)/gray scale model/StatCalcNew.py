import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
from variables import*

current_dir = os.getcwd()
csv_folder = 'data'
csv_dir = os.path.join(current_dir, csv_folder)

def get_values(df):
    data = df[inout_columns].values
    return data

def prepare_values(in_file, out_file, total_files):
    if (in_file in total_files) and (out_file in total_files):
        in_df = pd.read_csv('data\\'+in_file)
        out_df = pd.read_csv('data\\'+out_file)

        in_values = get_values(in_df)
        out_values = get_values(out_df)
        data_values = np.concatenate([in_values,out_values])

    elif (in_file in total_files) and (out_file not in total_files):
        in_df = pd.read_csv('data\\'+in_file)
        data_values = get_values(in_df)

    elif (in_file not in total_files) and (out_file in total_files):
        out_df = pd.read_csv('data\\'+out_file)
        data_values = get_values(out_df)

    data_values = data_values.reshape(-1,)
    if len(data_values) >= input_size:
        data_values = data_values[:input_size]
    else:
        data_values = np.array(data_values.tolist() + [0] * (input_size - len(data_values)))
    
    assert len(data_values) == input_size, "data vector should contain 784 components"
    return data_values

source_dir = Path(csv_dir)

in_files = [str(file).split(csv_dir)[1].split('.in')[0] for file in source_dir.glob('**/*.in.csv')]
out_files = [str(file).split(csv_dir)[1].split('.out')[0] for file in source_dir.glob('**/*.out.csv')]

csv_files = list(set(in_files + out_files))
csv_files.sort(key = lambda x: x.split('\\')[-1].split('.')[0])

total_files = [str(file).split(csv_dir)[1] for file in source_dir.glob('**/*.csv')]
data = []

for csv_index in csv_files:
    in_file = '.'.join([csv_index,'in','csv'])
    out_file = '.'.join([csv_index,'out','csv'])
    activity = {'activity': csv_index.split('\\')[-1]}

    values = prepare_values(in_file, out_file, total_files)

    cols = np.arange(1,input_size+1)
    file_data = {**activity, **dict(zip(cols, values))}
    data.append(file_data)

final_df = pd.DataFrame(data)
final_df.dropna(subset=final_df.columns.tolist()[1:], how='all', inplace=True)
final_df.fillna(0, inplace=True)
final_df['activity'] = final_df['activity'].apply(lambda x:re.sub(r'^([a-zA-Z]+).*', r'\1',x))
final_df.to_csv(os.path.join(current_dir,'NewData.csv') , index=False)