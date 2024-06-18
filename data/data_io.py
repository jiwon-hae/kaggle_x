import pandas as pd

from data.data_processing import *


def read_csv(file_path):
    return pd.read_csv(file_path)


def load_data(data_type: str):
    assert data_type == 'train' or data_type == 'test'
    if data_type == 'train':
        df = read_csv('resources/train.csv')
    else:
        df = read_csv('resources/test.csv')

    df = process_data(df)
    data_y = df['price']
    data_x = df.drop(columns=['price'])
    return data_x, data_y
