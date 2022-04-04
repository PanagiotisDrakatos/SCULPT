import os
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def rnn_data(data, time_steps, labels=False):
    """
        '''
            Creates all the possible sub-arrays (consisting of consequent array elements) of size "time_steps".
            This means that the model will `learn` this sequences, and we can hence use the first element of the
            time_steps-long sequence to predict/remember the last one.
        '''
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]] #Data frame for input with 2 timesteps
        -> labels == True [3, 4, 5] # labels for predicting the next timestep
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].to_numpy())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].to_numpy()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)

def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts; defaulting in 80-10-10
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[:ntest]

    return df_train, df_val, df_test

def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.5):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell. (lstm cells use ordinary rnn data)
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))

def read_input_files(working_dir):
    '''
        #Read the data from the input files (column indexing starts from zero)
        #time -> This is a timestamp of the record.
        #meas_info -> This is the site of the cell tower (antenna).
        #             You can think of this as a part of the telco's antenna with a specific range.
        #counter -> This is a number that represents a predefined KPI (Key Performance Indicator).
        #value  -> Is the value of the KPI.
    :param working_dir: directory in which the input files are in
    :return: rawdata
    '''
    rawdata = None
    df_list = []
    for root, dirs, files in os.walk(working_dir):
        file_list = []

        for filename in files:
            if filename.endswith('.csv'):
                file_list.append(os.path.join(root, filename))
        for file in file_list:
            df = pd.read_csv(file, delimiter="|", usecols=[1, 2, 6, 7], header=None, na_values=["NIL"],
                             na_filter=True, names=["meas_info", "counter", "value", "time"], index_col='time')
            df = df[df["counter"] == 67194794]  # extract data for this counter only
            df_list.append(df[["value"]])

    if df_list:
        rawdata = pd.concat(df_list)

    return rawdata


def load_csvdata(working_dir, time_steps, seperate=False):
    '''Separates the data into train, validation and test sets, and returns them'''
    data = read_input_files(working_dir)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)



def read_input_files_cell(working_dir, cell_name):
    '''
    #Read the data from the input files (column indexing starts from zero)
        #meas_info -> This is the site of the cell tower (antenna).
        #             You can think of this as a part of the telco's antenna with a specific range.
        #cell_name -> The name of the cell tower.
        #counter -> This is a number that represents a predefined KPI (Key Performance Indicator).
        #value  -> Is the value of the KPI.
        #time -> This is a timestamp of the record.
    :param working_dir: the directory of the input files
    :param cell_name: the name of the cell for whose data will be retrieved
    :return: rawdata
    '''
    rawdata = None
    df_list = []
    for root, dirs, files in os.walk(working_dir):
        file_list = []

        for filename in files:
            if filename.endswith('.csv'):
                file_list.append(os.path.join(root, filename))
        for file in file_list:
            df = pd.read_csv(file, delimiter="|", usecols=[1, 2, 5, 6, 7], header=None, na_values=["NIL"],
                             na_filter=True, names=["meas_info", "counter", "cellname", "value", "time"],
                             index_col='time')
            df = df[df["cellname"].str.contains(cell_name)]
            if not df.empty:
                df_list.append(df[["value"]])

    if df_list:
        rawdata = pd.concat(df_list)


    return rawdata




def load_csvdata_cell(working_dir, cell_name, time_steps, seperate=False):
    '''Separates the data into train, validation and test sets, and returns them'''
    data = read_input_files_cell(working_dir, cell_name)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)




def find_all_cell_names(working_dir):
    for root, dirs, files in os.walk(working_dir):
        file_list = []

        for filename in files:
            if filename.endswith('.csv'):
                file_list.append(os.path.join(root, filename))
        df_cells_list = []
        for file in file_list:
            df_cells = pd.read_csv(file, delimiter="|", usecols=[5, 7], header=None, na_values=["NIL"],
                                   na_filter=True, names=["cellname", "time"], index_col='time')
            df_cells_list.append(df_cells["cellname"].tolist())

        if df_cells_list:
            cells = np.unique(df_cells_list)

    return cells