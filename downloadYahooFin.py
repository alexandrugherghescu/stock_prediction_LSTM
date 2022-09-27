#!/usr/local/bin/python3
import yfinance as yf
import numpy as np
import pandas as pd

from datetime import datetime
from datetime import timedelta
from os import path
from os import remove


class DownloadFinancialData:

    #

    def download_data(self, ticker='VOO', interval='1h', output_path='', force_download=False):
        """
        Loads data from Yahoo Finance source, save it to data_in/<ticker>.csv . Also keep previous data if available.
        Params:
            ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
            interval (str): the interval for download, default is '1h'
            force_download (bool): force a download, default is False
        Out:
            <ticker>.csv
        """
        #str_file_name = path.join('data_in', f'{ticker}.csv')
        str_file_name = output_path
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
            dat_date_last = datetime.utcfromtimestamp(flt_date_seconds_last)
            if interval == '1h':
                # for 1h tick we limit to 730 days
                dat_start_date = dat_date_last - timedelta(days=2)
                dat_end_date = datetime.now().date() + timedelta(days=1)  # to avoid issues with time_delta relative to GMT
            else:
                # for 1d tick we limit to 10 years
                dat_start_date = dat_date_last - timedelta(days=10)
                dat_end_date = datetime.now().date() + timedelta(days=1)  # to avoid issues with time_delta relative to GMT
            dat_date_last_old = dat_date_last
            # this is first time --> we download max amount.
            # load it from yahoo_fin library
            df_in_new = yf.download(tickers=ticker, start=dat_start_date, end=dat_end_date, interval=interval)
            df_in_new.to_csv(str_file_name_temp)
            df_in_new = pd.read_csv(str_file_name_temp, sep=',')
            # convert to datetime64[ns]
            df_in_new['Date'] = pd.to_datetime(df_in_new['Date'], utc=True)
            # combine old and new data (drop duplicates)
            df_in = pd.concat([df_in, df_in_new], ignore_index=True).drop_duplicates(['Date'], keep='last')

            dat64_date_max = np.datetime64(df_in['Date'].max())
            # print(dat64DateMax)
            fltDateSecondsLast = (dat64_date_max - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            # convert to datetime
            dat_date_last = datetime.utcfromtimestamp(fltDateSecondsLast)

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
            # load it from yahoo_fin library
            df_in = yf.download(tickers=ticker, start=dat_start_date, end=dat_end_date, interval=interval)  # last 730 days
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
