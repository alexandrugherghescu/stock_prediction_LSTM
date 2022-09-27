#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#from dataclasses import dataclass, field
#from multiprocessing import cpu_count, Pool
#from pathlib import Path
#from time import perf_counter
#from typing import List, Tuple
#from warnings import simplefilter

#import pandas as pd
#from numpy import log10 as npLog10
#from numpy import ndarray as npNdarray
#from pandas.core.base import PandasObject

#from pandas_ta import Category, Imports, version
#from pandas_ta.candles.cdl_pattern import ALL_PATTERNS
#from pandas_ta.candles import *
#from pandas_ta.cycles import *
#from pandas_ta.momentum import *
from pandas_ta.overlap import *
#from pandas_ta.performance import *
#from pandas_ta.statistics import *
#from pandas_ta.trend import *
#from pandas_ta.volatility import *
#from pandas_ta.volume import *
#from pandas_ta.utils import *





#import yfinance as yf
import numpy as np
import pandas as pd
#import pandas_ta as ta

from datetime import datetime
from datetime import timedelta
from os import path
from os import remove


class Merge:

    #

    def merge_data(self, ticker, future_steps=5, input_path='', processed_path='', output_path=''):
        """
        Merge ticker with output_pred_X.csv.
        Params:
            ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
            future_steps (int): the prediction length
            input_path (str): the input file that need to be merged
            output_path (str): the output file that need to be merged
        Out:
            ticker_pred_X.csv
        """

        #strFileName = path.join('data_in', f'{ticker}.csv')
        # load from CSVs
        obj_data_frame_input = pd.read_csv(input_path, sep=',')
        obj_data_frame = pd.read_csv(processed_path, sep=',')
        # shift data down 'future_steps'
        str_date_last = obj_data_frame['Date'].iloc[-1]
        dat_date_last = datetime.strptime(str_date_last, '%Y-%m-%d')        #  '%d/%m/%y %H:%M:%S'
        str_dic_date = []
        for intA in range(future_steps):
            dat_date = dat_date_last + timedelta(days=(intA+1))
            str_dic_date.append(datetime.strftime(dat_date, '%Y-%m-%d'))
        for strDate in str_dic_date:
            # ('Date,Loss,MAE,Epochs,Prediction\n')
            obj_data_frame.loc[len(obj_data_frame.index)] = [strDate, np.nan, np.nan, np.nan, np.nan]
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
    #    strFileName = path.join('data_proc', f'{ticker}_{future_steps}_final.csv')
        obj_data_frame_input.to_csv(output_path, index=False)