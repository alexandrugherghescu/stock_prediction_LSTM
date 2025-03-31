#!/usr/local/bin/python3

import pandas as pd
import pandas_ta as ta
from os import path
from os import mkdir
from commonResources import *


class FinancialParams:


    def setup_folders(self):
        """
        Setup folder data_in for download (if not already available)
        Params:
            None
        Out:
            None
        """
        # create folders if they don't exist
        if not path.isdir(FOLDER_NAME_FOR_DATA_FINANCIAL):
            # for data input (csv)
            mkdir(FOLDER_NAME_FOR_DATA_FINANCIAL)


    def add_financial_data(self, ticker, arr_LOOKUP_STEP=[12, 24, 48, 64, 96, 144, 192]):
        """
        Add different financial parameters, save it to data_financial/<ticker>_param.csv .
        Params:
            ticker (str): the ticker to get the file name
            arr_LOOKUP_STEP (int) list: number of column that we plan to use for training
        Out:
            <ticker>_param.csv
            return length of the dataset (to be used on next stage)
        """

        str_file_name_in = path.join(FOLDER_NAME_FOR_DATA_IN, f'{ticker}.csv')
        str_file_name_out = path.join(FOLDER_NAME_FOR_DATA_FINANCIAL, f'{ticker}_param.csv')

        obj_data_frame = pd.read_csv(str_file_name_in, sep=',')
        # add MOMentum
        obj_data_frame.ta.mom(length=1, append=True)
        obj_data_frame.ta.mom(length=3, append=True)
        obj_data_frame.ta.mom(length=5, append=True)
        obj_data_frame.ta.mom(length=10, append=True)
        obj_data_frame.ta.mom(length=20, append=True)
        # add Rate Of Change
        obj_data_frame.ta.roc(length=2, append=True)
        obj_data_frame.ta.roc(length=5, append=True)
        obj_data_frame.ta.roc(length=10, append=True)
        obj_data_frame.ta.roc(length=20, append=True)
        obj_data_frame.ta.roc(length=40, append=True)
        # add Exponential Moving Average
        obj_data_frame.ta.ema(length=5, append=True)
        obj_data_frame.ta.ema(length=10, append=True)
        obj_data_frame.ta.ema(length=20, append=True)
        obj_data_frame.ta.ema(length=40, append=True)
        # add STandard DEViation
        obj_data_frame.ta.stdev(length=3, append=True)
        obj_data_frame.ta.stdev(length=5, append=True)
        obj_data_frame.ta.stdev(length=8, append=True)
        obj_data_frame.ta.stdev(length=15, append=True)
        # add diffs (derivative of parameters)
        obj_data_frame['dMOM_1'] = obj_data_frame['MOM_1'].diff()
        obj_data_frame['dMOM_3'] = obj_data_frame['MOM_3'].diff()
        obj_data_frame['dMOM_5'] = obj_data_frame['MOM_5'].diff()
        obj_data_frame['dMOM_10'] = obj_data_frame['MOM_10'].diff()
        obj_data_frame['dMOM_20'] = obj_data_frame['MOM_20'].diff()
        obj_data_frame['dROC_2'] = obj_data_frame['ROC_2'].diff()
        obj_data_frame['dROC_5'] = obj_data_frame['ROC_5'].diff()
        obj_data_frame['dROC_10'] = obj_data_frame['ROC_10'].diff()
        obj_data_frame['dROC_20'] = obj_data_frame['ROC_20'].diff()
        obj_data_frame['dROC_40'] = obj_data_frame['ROC_40'].diff()
        obj_data_frame['dEMA_5'] = obj_data_frame['EMA_5'].diff()
        obj_data_frame['dEMA_10'] = obj_data_frame['EMA_10'].diff()
        obj_data_frame['dEMA_20'] = obj_data_frame['EMA_20'].diff()
        obj_data_frame['dEMA_40'] = obj_data_frame['EMA_40'].diff()
        obj_data_frame['dSTDEV_3'] = obj_data_frame['STDEV_3'].diff()
        obj_data_frame['dSTDEV_5'] = obj_data_frame['STDEV_5'].diff()
        obj_data_frame['dSTDEV_8'] = obj_data_frame['STDEV_8'].diff()
        obj_data_frame['dSTDEV_15'] = obj_data_frame['STDEV_15'].diff()
        # calculate output: this is required for training
        # we use Close as target shifted with 'lookup_step' steps
        # loop through all parameters and generate one PRED_* column for each prediction range
        for lookup_step in arr_LOOKUP_STEP:
            # shift backwards
            obj_data_frame['PRED_' + str(lookup_step)] = obj_data_frame['Close'].shift(-lookup_step)
        # save to CSV file
        obj_data_frame.to_csv(str_file_name_out, index=False)
        return len(obj_data_frame)


    def get_dataframe_length(self, ticker):
        """
        Return the length of data_financial/<ticker>_param.csv .
        Params:
            ticker (str): the ticker to get the file name
        Out:
            return length of the dataset (to be used on next stage)
        """

        str_file_name_out = path.join(FOLDER_NAME_FOR_DATA_FINANCIAL, f'{ticker}_param.csv')
        obj_data_frame = pd.read_csv(str_file_name_out, sep=',')
        return len(obj_data_frame)
