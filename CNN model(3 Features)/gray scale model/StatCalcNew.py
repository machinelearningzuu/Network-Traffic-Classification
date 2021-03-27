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

        if len(in_values) > len(out_values):
            in_values = in_values[np.random.choice(len(in_values), len(out_values), replace=False)]
        elif len(in_values) < len(out_values):
            out_values = out_values[np.random.choice(len(out_values), len(in_values), replace=False)]
        
        data_values = np.concatenate([in_values,out_values], axis=1)
        data_csv = data_values

    elif (in_file in total_files) and (out_file not in total_files):
        in_df = pd.read_csv('data\\'+in_file)
        data_values = get_values(in_df)
        N, K = data_values.shape
        data_csv = np.concatenate([data_values,np.zeros((N, K))], axis=1)

    elif (in_file not in total_files) and (out_file in total_files):
        out_df = pd.read_csv('data\\'+out_file)
        data_values = get_values(out_df)
        N, K = data_values.shape
        data_csv = np.concatenate([np.zeros((N, K)), data_values], axis=1)

    data_values = np.resize(data_values,(input_shape[0],input_shape[1]))
    data_values = data_values.reshape(-1,)

    assert len(data_values) == input_size, "data vector should contain 784 components"
    return data_values,data_csv

source_dir = Path(csv_dir)

in_files = [str(file).split(csv_dir)[1].split('.in')[0] for file in source_dir.glob('**/*.in.csv')]
out_files = [str(file).split(csv_dir)[1].split('.out')[0] for file in source_dir.glob('**/*.out.csv')]

csv_files = list(set(in_files + out_files))
csv_files.sort(key = lambda x: x.split('\\')[-1].split('.')[0])

total_files = [str(file).split(csv_dir)[1] for file in source_dir.glob('**/*.csv')]
data = []

for i, csv_index in enumerate(csv_files):
    in_file = '.'.join([csv_index,'in','csv'])
    out_file = '.'.join([csv_index,'out','csv'])
    act = csv_index.split('\\')[-1].split('.')[0][:-1]
    activity = {'activity': act}
    if len(activity) > 0:
        values, csv_values = prepare_values(in_file, out_file, total_files)

        cols = np.arange(1,input_size+1)
        file_data = {**activity, **dict(zip(cols, values))}
        csv_file_data = np.concatenate([np.array([act] * len(csv_values)).reshape(-1,1), csv_values], axis=1)

        if i==0:
            data_csv = csv_file_data
        else:
            data_csv = np.concatenate([data_csv, csv_file_data])

        data.append(file_data)

final_df = pd.DataFrame(data)
final_df.dropna(subset=final_df.columns.tolist()[1:], how='all', inplace=True)
final_df.fillna(0, inplace=True)
final_df.to_csv(os.path.join(current_dir,train_csv) , index=False)

# vis_df = pd.DataFrame(data=data_csv, columns=vis_inout_columns)
# vis_df.dropna(subset=vis_df.columns.tolist()[1:], how='all', inplace=True)
# vis_df.fillna(0, inplace=True)
# vis_df.to_csv(os.path.join(current_dir,'visualize_new_data.csv') , index=False)