#!/usr/local/bin/python3
import pandas as pd

from os import path


class ProcessData:

    #
    def process_split_data(self, ticker, test_size=0.2):
        """
        Loads data from CSV file and split into Train / Test set.
        Params:
            ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
            test_size (float): ratio for test data, default is 0.2 (20% testing data)
        Out:
            ticker_train.csv
            ticker_test.csv
        """
        #
        # load it from data/ticker_param.csv
        str_file_name_in = path.join('data_financial', f'{ticker}_param.csv')
        str_file_name_train = path.join('data_proc', f'{ticker}_train.csv')
        str_file_name_test = path.join('data_proc', f'{ticker}_test.csv')
        df = pd.read_csv(str_file_name_in, sep=',')

        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(df))
        train = df[:train_samples]
        test = df[train_samples:]
        train.to_csv(str_file_name_train, index=False)
        test.to_csv(str_file_name_test, index=False)

    def process_split_data_by_index(self, ticker, index=1, overlap=1, training_length=0, testing_length=0):
        """
        Loads data from CSV file - using index
        Params:
            ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
            index (int): number where we stop the training (index in buffer)
            overlap (int): how many samples we overlap between train and test set (due to network length).
            training_length (int): how many samples is the training set
            testing_length (int): how many samples is the testing set
        Out:
            ticker_train.csv
            ticker_test.csv
        """
        #
        # load it from data/ticker_param.csv
        str_file_name_in = path.join('data_financial', f'{ticker}_param.csv')
        str_file_name_train = path.join('data_proc', f'{ticker}_train.csv')
        str_file_name_test = path.join('data_proc', f'{ticker}_test.csv')
        df = pd.read_csv(str_file_name_in, sep=',')

        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int(index)
        train = df[:train_samples]
        # check if we limit number of samples
        if training_length > 0:
            train = train.tail(training_length)
        test = df[train_samples-overlap:]
        # check if we limit number of samples
        if testing_length > 0:
            test = test.head(testing_length+overlap)
        # save to csv
        train.to_csv(str_file_name_train, index=False)
        test.to_csv(str_file_name_test, index=False)

    def get_date_at_index(self, ticker, index=1, overlap=1, training_length=0, testing_length=0):
        """
        Loads data from CSV file - using index
        Params:
            ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
            index (int): number where we stop the training (index in buffer)
            overlap (int): how many samples we overlap between train and testset (due to network length).
            training_length (int): how many samples is the training set
            testing_length (int): how many samples is the testing set
        Out:
            str_date_last (str): date from the index
        """
        #
        # load it from data/ticker_param.csv
        str_file_name_in = path.join('data_financial', f'{ticker}_param.csv')
        df = pd.read_csv(str_file_name_in, sep=',')
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int(index)
        train = df[:train_samples]
        # check if we limit number of samples
        if training_length > 0:
            train = train.tail(training_length)
        test = df[train_samples - overlap:]
        # check if we limit number of samples
        if testing_length > 0:
            test = test.head(testing_length + overlap)
        str_date_last = test['Date'].iloc[-1]
        return str_date_last

