#!/usr/local/bin/python3
import pandas as pd
import numpy as np

#from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Activation
from tensorflow.keras.models import Sequential
from os import path
from os import remove
from commonResources import *
from pathlib import Path
from joblib import dump, load

class ML:


    def create_model(self, sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
        model = Sequential()
        # we build a model with several layers.
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
                else:
                    model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="linear"))
        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer, run_eagerly=True)
        return model


    def predict(self, model, data, scale, scaler):
        # expand dimension
        last_sequence = np.expand_dims(data, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)

        # get the price
        #predicted_price = prediction[0][0]
        if scale:
            predicted_price = scaler.inverse_transform(prediction)[0][0]
        else:
            predicted_price = prediction[0][0]
        return predicted_price


    def prediction_to_csv(self, ticker, sequence_length=50, future_steps=24, neurons=256, network_layers=3, drop_out=0.4, bidirectional=False, test_size=0.2, epoch=1, FEATURE_COLUMNS=[], scale=False, MINMAX_COLUMNS=[], STANDARD_COLUMNS=[]):
        # Window size or the sequence length
        INPUT_WINDOW_SIZE = sequence_length
        # Lookup step, 1 is the next day
        LOOKUP_STEP = future_steps

        ### model parameters ###
        N_LAYERS = network_layers
        # LSTM cell
        CELL = LSTM
        # 256 LSTM neurons
        UNITS = neurons
        # 40% dropout
        DROPOUT = drop_out
        # whether to use bidirectional RNNs
        BIDIRECTIONAL = bidirectional

        ### training parameters ###
        # mean absolute error loss
        # LOSS = "mae"
        # huber loss
        LOSS = "huber_loss"
        OPTIMIZER = "adam"

        # model name to save, making it as unique as possible based on parameters
        model_name = f"{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{INPUT_WINDOW_SIZE}-lookup-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}-dropout-{DROPOUT}"
        if BIDIRECTIONAL:
            model_name += "-b"

        # load the data
        str_file_name = path.join(FOLDER_NAME_FOR_DATA_PROCESSED, f'{ticker}_test.csv')
        df_test = pd.read_csv(str_file_name, sep=',')
        # copy data where we append results as a column
        df_out = df_test.copy(deep=True)

        if scale:
            # load scaler
            # column_scaler = {}
            column_scaler = load('scalerX.bin')
            # scale the data (prices) from 0 to 1
            for column in FEATURE_COLUMNS:
                scaler = column_scaler[column]
                df_test[column] = scaler.transform(np.expand_dims(df_test[column].values, axis=1))

        # construct the model
        model = self.create_model(INPUT_WINDOW_SIZE, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                            dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

        # load optimal model weights from results folder
        model_path = path.join(FOLDER_NAME_FOR_RESULTS, model_name) + ".h5"
        model.load_weights(model_path)

        # select only columns=FEATURES last 20%
        int_df_length = len(df_test)
        int_buffer_length = int(int_df_length * test_size)
        int_buffer_length = int_buffer_length + 7
        df_test = df_test[FEATURE_COLUMNS]
        # remove NANs
        df_test.dropna(inplace=True)
        lst_output = []
        # load scaler
        scaler = load('scalery'  + str(LOOKUP_STEP) + '.bin')
        # fill in output with NANs for 80% + network length
        #int_output_length = int_df_length - int_buffer_length + INPUT_WINDOW_SIZE - 1
        int_output_length = INPUT_WINDOW_SIZE - 1
        for intA in range(int_output_length):
            lst_output.append(np.nan)
        # fill in output with predictions
        for intA in range(len(df_test)-INPUT_WINDOW_SIZE+1):
            data = df_test.tail(len(df_test)-intA)
            data = data.head(INPUT_WINDOW_SIZE)
            data = np.array(data)
            # predict the future price
            future_price = self.predict(model, data, scale, scaler)
            lst_output.append(future_price)
        # save to temp file
        np.savetxt("temp.csv", lst_output, delimiter=",")
        df_predict = pd.read_csv("temp.csv", header = None)
        df_predict.columns = ['CALC_' + str(INPUT_WINDOW_SIZE) + '_' + str(LOOKUP_STEP)]
        # append calculated column
        df_out = pd.concat([df_out, df_predict], axis=1)
        # save to csv file
        str_file_name = path.join('data', f'{model_name}-epoch-{epoch}_final.csv')
        df_out.to_csv(str_file_name, index=False)
        # delete temp files
        my_file = Path('temp.csv')
        if my_file.is_file():
            remove('temp.csv')
        #print(model.summary())
        print("***************************************************************")


    def last_prediction_from_train_data(self, ticker, sequence_length=50, future_steps=24, neurons=256, network_layers=3, drop_out=0.4, bidirectional=False, FEATURE_COLUMNS=[], scale=False):
        # Window size or the sequence length
        INPUT_WINDOW_SIZE = sequence_length
        # Lookup step, 1 is the next day
        LOOKUP_STEP = future_steps

        ### model parameters ###
        N_LAYERS = network_layers
        # LSTM cell
        CELL = LSTM
        # 256 LSTM neurons
        UNITS = neurons
        # 40% dropout
        DROPOUT = drop_out
        # whether to use bidirectional RNNs
        BIDIRECTIONAL = bidirectional

        ### training parameters ###
        # mean absolute error loss
        # LOSS = "mae"
        # huber loss
        LOSS = "huber_loss"
        OPTIMIZER = "adam"

        # model name to save, making it as unique as possible based on parameters
        model_name = f"{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{INPUT_WINDOW_SIZE}-lookup-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}-dropout-{DROPOUT}"
        if BIDIRECTIONAL:
            model_name += "-b"

        # load the data
        str_file_name = path.join(FOLDER_NAME_FOR_DATA_PROCESSED, f'{ticker}_test.csv')
        df_test = pd.read_csv(str_file_name, sep=',')
        # copy data where we append results as a column
        str_date_last = df_test['Date'].iloc[-1]

        # model name to save, making it as unique as possible based on parameters
        model_name = f"{ticker}-{str_date_last}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{INPUT_WINDOW_SIZE}-lookup-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}-dropout-{DROPOUT}"
        if BIDIRECTIONAL:
            model_name += "-b"

        if scale:
            # load scaler
            str_file_name = path.join(FOLDER_NAME_FOR_SCALER, f'{model_name}_scalerX.bin')
            column_scaler = load(str_file_name)
            # scale the data (prices) from 0 to 1
            for column in FEATURE_COLUMNS:
                scaler = column_scaler[column]
                df_test[column] = scaler.transform(np.expand_dims(df_test[column].values, axis=1))

        # construct the model
        model = self.create_model(INPUT_WINDOW_SIZE, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                            dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

        # load optimal model weights from results folder
        model_path = path.join(FOLDER_NAME_FOR_RESULTS, model_name) + ".h5"
        model.load_weights(model_path)

        # select only columns=FEATURES last 20%
        df_test = df_test[FEATURE_COLUMNS]
        # remove NANs
        df_test.dropna(inplace=True)
        # load scaler
        str_file_name = path.join(FOLDER_NAME_FOR_SCALER, f'{model_name}_scalery_{LOOKUP_STEP}.bin')
        scaler = load(str_file_name)
        #
        data = df_test.tail(INPUT_WINDOW_SIZE)
        # predict the future price
        future_price = self.predict(model, data, scale, scaler)

        return str_date_last, future_price
