#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


from datetime import datetime
from datetime import timedelta
from pandas_ta.overlap import *
from os import path
from os import mkdir
from commonResources import *


class Merge:


    def setup_folders(self):
        """
        Setup folder data_in for download (if not already available)
        Params:
            None
        Out:
            None
        """
        # create folders if they don't exist
        if not path.isdir(FOLDER_NAME_FOR_DATA_FINAL):
            # for data input (csv)
            mkdir(FOLDER_NAME_FOR_DATA_FINAL)


    def merge_data(self, ticker, future_steps=5, str_file_name=''):
        """
        Merge ticker with output_pred_X.csv.
        Params:
            ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
            future_steps (int): the prediction length
            str_file_name (str): the root of file that need to be merged
        Out:
            ticker_pred_X.csv
        """

        #
        input_path = path.join(FOLDER_NAME_FOR_DATA_IN, f'{ticker}.csv')
        processed_path = path.join(FOLDER_NAME_FOR_DATA_PREDICTIONS, f'{str_file_name}_pred.csv')
        output_path = path.join(FOLDER_NAME_FOR_DATA_FINAL, f'{str_file_name}.csv')

        # load from CSVs
        obj_data_frame_input = pd.read_csv(input_path, sep=',')
        obj_data_frame = pd.read_csv(processed_path, sep=',')
        # shift data down 'future_steps'
        str_date_last = obj_data_frame['Date'].iloc[-1]
        dat_date_last = datetime.strptime(str_date_last, '%Y-%m-%d')        #  '%d/%m/%y %H:%M:%S'
        str_dic_date = []
        for int_a in range(future_steps):
            dat_date = dat_date_last + timedelta(days=(int_a+1))
            str_dic_date.append(datetime.strftime(dat_date, '%Y-%m-%d'))
        for str_date in str_dic_date:
            # ('Date,Loss,MAE,Epochs,Prediction\n')
            obj_data_frame.loc[len(obj_data_frame.index)] = [str_date, np.nan, np.nan, np.nan, np.nan]
        # shift forwards
        obj_data_frame['Prediction'] = obj_data_frame['Prediction'].shift(future_steps)
        # merge frames
        obj_data_frame_input = pd.merge(obj_data_frame_input, obj_data_frame, on='Date', how='outer')
        # calculate error between Close and Prediction
        obj_data_frame_input['Error'] = obj_data_frame_input['Close'] - obj_data_frame_input['Prediction']
        # calculate relative MAE
        obj_data_frame_input['rMAE'] = obj_data_frame_input['MAE'] / obj_data_frame_input['Close']
        # get the MIN and MAX to calculate the confidence level --> 0: low ... 1: high
        # remove pedestal
        flt_rMAE_min = obj_data_frame_input['rMAE'].min()
        obj_data_frame_input['rMAE'] = obj_data_frame_input['rMAE'] - flt_rMAE_min
        # create ration 0-->1
        flt_rMAE_max = obj_data_frame_input['rMAE'].max()
        obj_data_frame_input['rMAE'] = obj_data_frame_input['rMAE'] / flt_rMAE_max
        # invert ratio
        obj_data_frame_input['rMAE'] = 1 - obj_data_frame_input['rMAE']
        # calculate abs()
        obj_data_frame_input['AbsError'] = abs(obj_data_frame_input['Error'])
        # calculate EMA 10 of Error
        obj_data_frame_input['EMA5AbsError'] = ema(obj_data_frame_input['AbsError'], length=5, offset=None, append=True)

        # calculate lower range for prediction
        obj_data_frame_input['PredLow'] = obj_data_frame_input['Prediction'] - obj_data_frame_input[['EMA5AbsError', 'Error']].max(axis=1)       # max(abs(objDataFrame['ErrorEMA5']),abs(objDataFrame['Error']))
        # calculate upper range for prediction
        obj_data_frame_input['PredHigh'] = obj_data_frame_input['Prediction'] + obj_data_frame_input[['EMA5AbsError', 'Error']].max(axis=1)      # max(abs(objDataFrame['ErrorEMA5']),abs(objDataFrame['Error']))
        # if abs(prediction - true) > true * 1% --> limit the wrong side
        flt_one_percent_limit = 0.01
        DEFAULT_VALUE = 0
        NO_CHANGE_ONE_PERCENT = 1
        CHANGE_LOW_VALUE = 2
        CHANGE_HIGH_VALUE = 3
        int_last_correction = DEFAULT_VALUE   # default
        for index, row in obj_data_frame_input.iterrows():
            if not pd.isna(obj_data_frame_input['Close'].values.item(index)):
                if abs(obj_data_frame_input['Close'].values.item(index) - obj_data_frame_input['Prediction'].values.item(index)) > (flt_one_percent_limit * obj_data_frame_input['Close'].values.item(index)):
                    if obj_data_frame_input['Close'].values.item(index) > obj_data_frame_input['Prediction'].values.item(index):
                        obj_data_frame_input.at[index, 'PredLow'] = obj_data_frame_input['Prediction'].values.item(index)
                        int_last_correction = CHANGE_LOW_VALUE   # change Low value
                    else:
                        obj_data_frame_input.at[index, 'PredHigh'] = obj_data_frame_input['Prediction'].values.item(index)
                        int_last_correction = CHANGE_HIGH_VALUE  # change High value
                else:
                    int_last_correction = NO_CHANGE_ONE_PERCENT   # in 1 % (no change)
            else:
                # change last future_steps values
                if int_last_correction == CHANGE_LOW_VALUE:
                    obj_data_frame_input.at[index, 'PredLow'] = obj_data_frame_input['Prediction'].values.item(index)
                elif int_last_correction == CHANGE_HIGH_VALUE:
                    obj_data_frame_input.at[index, 'PredHigh'] = obj_data_frame_input['Prediction'].values.item(index)

        # save df
    #    strFileName = path.join(FOLDER_NAME_FOR_DATA_PROCESSED, f'{ticker}_{future_steps}_final.csv')
        obj_data_frame_input.to_csv(output_path, index=False)