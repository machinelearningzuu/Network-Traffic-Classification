import re
import os
import pandas as pd
from pathlib import Path

current_dir = os.getcwd()
csv_folder = 'Dataset'
csv_dir = os.path.join(current_dir, csv_folder)

def prepare_values(df):
    df_columns = ['frame.time_delta_displayed', 'frame.len']
    df_values = []
    for col in df_columns:
        df_values +=[
            df[col].max(),
            df[col].min(),
            df[col].std(),
            df[col].quantile(0.25),
            df[col].quantile(0.5),
            df[col].quantile(0.75),
            df[col].mean(),
            df[col].mad(),
            df[col].var(),
            df[col].skew(),
            df[col].kurtosis(),
            df[col].sum(),
        ]
    return df_values

source_dir = Path(csv_dir)

# in_files = [os.path.split(file)[1].split('.in.csv')[0] for file in source_dir.glob('**/*.in.csv')]
# out_files = [os.path.split(file)[1].split('.out.csv')[0] for file in source_dir.glob('**/*.out.csv')]

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
    in_cols = ['maxTimeIn', 'minTimeIn', 'stdTimeIn', 'q1TimeIn', 'q2TimeIn', 'q3TimeIn', 'meanTimeIn', 'madTimeIn', 'varianceTimeIn', 'skewTimeIn', 'kurtosisTimeIn', 'sumTimeIn', 'maxLenIn', 'minLenIn', 'stdLenIn', 'q1LenIn','q2lenIn', 'q3LenIn', 'meanLenIn', 'madLenIn', 'varianceLenIn', 'skewLenIn', 'kurtosisLenIn', 'sumLenIn']
    out_cols = ['maxTimeOut', 'minTimeOut', 'stdTimeOut', 'q1TimeOut', 'q2TimeOut', 'q3TimeOut', 'meanTimeOut', 'madTimeOut', 'varianceTimeOut', 'skewTimeOut', 'kurtosisTimeOut', 'sumTimeOut', 'maxLenOut', 'minLenOut', 'stdLenOut', 'q1LenOut', 'q2LenOut', 'q3LenOut', 'meanLenOut', 'madLenOut', 'varianceLenOut', 'skewLenOut', 'kurtosisLenOut','sumLenOut']

    if (in_file in total_files) and (out_file in total_files):
        in_df = pd.read_csv('Dataset\\'+in_file)
        out_df = pd.read_csv('Dataset\\'+out_file)

        in_values = prepare_values(in_df)
        out_values = prepare_values(out_df)

    elif (in_file in total_files) and (out_file not in total_files):
        in_df = pd.read_csv('Dataset\\'+in_file)

        in_values = prepare_values(in_df)
        out_values = [0]*len(in_values)

    elif (in_file not in total_files) and (out_file in total_files):
        out_df = pd.read_csv('Dataset\\'+out_file)

        out_values = prepare_values(out_df)
        in_values = [0]*len(out_values)

    values = in_values + out_values
    cols = in_cols + out_cols

    file_data = {**activity, **dict(zip(cols, values))}
    data.append(file_data)

final_df = pd.DataFrame(data)
final_df.dropna(subset=final_df.columns.tolist()[1:], how='all', inplace=True)
final_df.fillna(0, inplace=True)
final_df['activity'] = final_df['activity'].apply(lambda x:re.sub(r'^([a-zA-Z]+).*', r'\1',x))
final_df.to_csv(os.path.join(current_dir,'NewData.csv') , index=False)