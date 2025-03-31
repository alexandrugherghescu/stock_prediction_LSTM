#!/usr/local/bin/python3

import pandas_datareader.data as web
import numpy as np
import pandas as pd

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from os import path
from os import remove
from os import mkdir
from commonResources import *


class DownloadFinancialData:

    def setup_folders(self):
        """
        Setup folder data_in for download (if not already available)
        Params:
            None
        Out:
            None
        """
        # create folders if they don't exist
        if not path.isdir(FOLDER_NAME_FOR_DATA_IN):
            # for data input (csv)
            mkdir(FOLDER_NAME_FOR_DATA_IN)


    def download_data(self, ticker='VOO', interval='1h', force_download=False):
        """
        Loads data from Stooq Finance source, save it to data_in/<ticker>.csv . Also keep previous data if available.
        Params:
            ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
            interval (str): the interval for download, default is '1h'
            force_download (bool): force a download, default is False
        Out:
            <ticker>.csv
        """
        str_file_name = path.join(FOLDER_NAME_FOR_DATA_IN, f'{ticker}.csv')
        #str_file_name = output_path
        str_file_name_temp = path.join('temp', f'temp.csv')
        bln_return = False

        # we try to reuse the CSV data downloaded previously
        if path.exists(str_file_name) and not force_download:
            # we already downloaded some data --> we have to append new data
            df_in = pd.read_csv(str_file_name, sep=',')
            # get last date
            # convert to datetime64[ns]
            df_in['Date'] = pd.to_datetime(df_in['Date'], utc=True)
            dat64_date_max = np.datetime64(df_in['Date'].max())
            #print(dat64DateMax)
            flt_date_seconds_last = (dat64_date_max - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            # convert to datetime
            dat_date_last = datetime.fromtimestamp(flt_date_seconds_last, timezone.utc)

            if interval == '1h':
                # for 1h tick we limit to 2 days
                dat_start_date = dat_date_last.date() - timedelta(days=2)
                dat_end_date = datetime.now().date() + timedelta(days=1)  # to avoid issues with time_delta relative to GMT
            else:
                # for 1d tick we limit to 10 days
                dat_start_date = dat_date_last.date() - timedelta(days=10)
                dat_end_date = datetime.now().date() + timedelta(days=1)  # to avoid issues with time_delta relative to GMT
            # save to determine if new data is available for processing
            dat_date_last_old = dat_date_last
            # this is patch download --> we download small amount.
            df_in_new = web.DataReader(ticker, 'stooq', start=dat_start_date, end=dat_end_date)
            df_in_new = df_in_new.sort_values(by='Date', ascending=True)
            df_in_new.to_csv(str_file_name_temp)
            df_in_new = pd.read_csv(str_file_name_temp, sep=',')
            # convert to datetime64[ns]
            df_in_new['Date'] = pd.to_datetime(df_in_new['Date'], utc=True)
            # combine old and new data (drop duplicates)
            df_in = pd.concat([df_in, df_in_new], ignore_index=True).drop_duplicates(['Date'], keep='last')

            dat64_date_max = np.datetime64(df_in['Date'].max())
            # print(dat64DateMax)
            flt_date_seconds_last = (dat64_date_max - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            # convert to datetime
            dat_date_last = datetime.fromtimestamp(flt_date_seconds_last, timezone.utc)

            # convert Date to string
            df_in['Date'] = df_in['Date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
            # save df
            df_in.to_csv(str_file_name, index=False)
            if dat_date_last_old != dat_date_last:
                bln_return = True
        else:
            if interval == '1h':
                # for 1h tick we limit to 730 days
                dat_start_date = datetime.now().date() - timedelta(days=728)
                dat_end_date = datetime.now().date() + timedelta(days=1)  # to avoid issues with time_delta relative to GMT
            else:
                # for 1d tick we limit to 5 years
                dat_start_date = datetime.now().date() - timedelta(days=1780)
                dat_end_date = datetime.now().date() + timedelta(days=1)  # to avoid issues with time_delta relative to GMT
            # this is first time --> we download max amount.
            df_in = web.DataReader(ticker, 'stooq', start=dat_start_date, end=dat_end_date)
            df_in = df_in.sort_values(by='Date', ascending=True)
            df_in.to_csv(str_file_name_temp)
            df_in = pd.read_csv(str_file_name_temp, sep=',')
            df_in.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
            # save df
            df_in.to_csv(str_file_name, index=False)
        # remove temp files
        if path.exists(str_file_name_temp):
            try:
                remove(str_file_name_temp)
            except OSError:
                pass

        return bln_return
