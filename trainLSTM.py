#!/usr/local/bin/python3
import pandas as pd
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from os import path
from collections import deque
from sklearn import preprocessing
from joblib import dump, load


class ML:
    # Implement callback function to stop training prematurely
    mean_absolute_error_last = 1000.0
    mean_absolute_error_counter = 0
    return_epochs = 0
    exit_if_no_improvement_for = 30

    class MyCallback(Callback):

        def on_epoch_end(self, epoch, logs={}):
            if logs.get('mean_absolute_error') is not None:
                mean_absolute_error = logs.get('mean_absolute_error')
                if mean_absolute_error < ML.mean_absolute_error_last:
                    ML.mean_absolute_error_counter = 0
                    ML.mean_absolute_error_last = mean_absolute_error
                else:
                    ML.mean_absolute_error_counter = ML.mean_absolute_error_counter + 1
                    if ML.mean_absolute_error_counter > ML.exit_if_no_improvement_for:
                        print("**********************************************************")
                        print("Reached steady mean_absolute_error, so stopping training!!")
                        print("**********************************************************")
                        self.model.stop_training = True
                        ML.return_epochs = epoch

    # Instantiate a callback object
    callbacks = MyCallback()

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

    def train_model(self, ticker, epochs_start=500, epochs_retrain=2, sequence_length=50, future_steps=24, neurons=256, network_layers=3, drop_out=0.4, bidirectional=False, FEATURE_COLUMNS=[], scale=False, MINMAX_COLUMNS=[], STANDARD_COLUMNS=[], testing_lenght=0, exit_if_no_improvement_for=30, allow_model_loading=True):
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
        BATCH_SIZE = 64
        EPOCHS = epochs_start
        ML.exit_if_no_improvement_for = exit_if_no_improvement_for

        # load the data
        # load it from data/ticker_param.csv
        str_file_name = path.join('data_proc', f'{ticker}_train.csv')
        df_train = pd.read_csv(str_file_name, sep=',')
        str_file_name = path.join('data_proc', f'{ticker}_test.csv')
        df_test = pd.read_csv(str_file_name, sep=',')
        # get last date to generate training model with last data add in name
        str_date_last = df_test['Date'].iloc[-1]
        # limit the test set
        if testing_lenght > 0:
            df_test = df_test.head(sequence_length + testing_lenght)
        # drop nan
        df_train.dropna(inplace=True)
        df_test.dropna(inplace=True)

        # filter FEATURE columns
        df_train_X = df_train[FEATURE_COLUMNS]
        df_test_X = df_test[FEATURE_COLUMNS]
        Y_COLUMN_NAME = 'PRED_' + str(LOOKUP_STEP)
        df_train_y = df_train[[Y_COLUMN_NAME]]
        df_test_y = df_test[[Y_COLUMN_NAME]]

        pd.options.mode.chained_assignment = None  # default='warn'
        # add date as a column
        if 'date' not in df_train_X.columns:
            df_train_X['date'] = df_train_X.index
        # add date as a column
        if "date" not in df_test_X.columns:
            df_test_X["date"] = df_test_X.index
        pd.options.mode.chained_assignment = 'warn'  # default='warn'

        # model name to save, making it as unique as possible based on parameters
        model_name = f"{ticker}-{str_date_last}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{INPUT_WINDOW_SIZE}-lookup-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}-dropout-{DROPOUT}"
        if BIDIRECTIONAL:
            model_name += "-b"

        if scale:
            column_scaler = {}
            # scale X data (prices) from 0 to 1 or -1 to 1
            for column in FEATURE_COLUMNS:
                if column in MINMAX_COLUMNS:
                    scaler = preprocessing.MinMaxScaler()
                elif column in STANDARD_COLUMNS:
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.StandardScaler()
                pd.options.mode.chained_assignment = None  # default='warn'
                df_train_X[column] = scaler.fit_transform(np.expand_dims(df_train_X[column].values, axis=1))
                df_test_X[column] = scaler.transform(np.expand_dims(df_test_X[column].values, axis=1))
                pd.options.mode.chained_assignment = 'warn'  # default='warn'
                column_scaler[column] = scaler
            # save the scaler
            str_file_name = path.join('scaler', f'{model_name}_scalerX.bin')
            dump(column_scaler, str_file_name, compress=True)
            # scale Y data
            scaler = preprocessing.MinMaxScaler()
            pd.options.mode.chained_assignment = None  # default='warn'
            df_train_y[Y_COLUMN_NAME] = scaler.fit_transform(np.expand_dims(df_train_y[Y_COLUMN_NAME].values, axis=1))
            df_test_y[Y_COLUMN_NAME] = scaler.transform(np.expand_dims(df_test_y[Y_COLUMN_NAME].values, axis=1))
            pd.options.mode.chained_assignment = 'warn'  # default='warn'
            # save the scaler
            str_file_name = path.join('scaler', f'{model_name}_scalery_{LOOKUP_STEP}.bin')
            dump(scaler, str_file_name, compress=True)

        sequence_data_train = []
        sequences = deque(maxlen=INPUT_WINDOW_SIZE)
        for entry, target in zip(df_train_X[FEATURE_COLUMNS + ["date"]].values, df_train_y[Y_COLUMN_NAME].values):
            sequences.append(entry)
            if len(sequences) == INPUT_WINDOW_SIZE:
                sequence_data_train.append([np.array(sequences), target])

        sequence_data_test = []
        sequences = deque(maxlen=INPUT_WINDOW_SIZE)
        for entry, target in zip(df_test_X[FEATURE_COLUMNS + ["date"]].values, df_test_y[Y_COLUMN_NAME].values):
            sequences.append(entry)
            if len(sequences) == INPUT_WINDOW_SIZE:
                sequence_data_test.append([np.array(sequences), target])

        # construct the X's and y's
        lst_train_X, lst_train_y = [], []
        for seq, target in sequence_data_train:
            lst_train_X.append(seq)
            lst_train_y.append(target)
        lst_test_X, lst_test_y = [], []
        for seq, target in sequence_data_test:
            lst_test_X.append(seq)
            lst_test_y.append(target)
        # convert to numpy arrays
        nparr_train_X = np.array(lst_train_X)
        nparr_train_y = np.array(lst_train_y)
        nparr_test_X = np.array(lst_test_X)
        nparr_test_y = np.array(lst_test_y)

        # remove dates from the training/testing sets & convert to float32
        nparr_train_X = nparr_train_X[:, :, :len(FEATURE_COLUMNS)].astype(np.float32)
        nparr_test_X = nparr_test_X[:, :, :len(FEATURE_COLUMNS)].astype(np.float32)

        # construct the model
        model = self.create_model(INPUT_WINDOW_SIZE, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                                  dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

        # some tensorflow callbacks
        checkpointer = ModelCheckpoint(path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
        tensorboard = TensorBoard(log_dir=path.join("logs", model_name))

        # train the model and save the weights whenever we see improvement
        if path.isdir("results"):
            # load optimal model weights from results folder
            model_path = path.join("results", model_name) + ".h5"
            if allow_model_loading and path.exists(model_path):
                # if file exist
                model.load_weights(model_path)
                # overwrite the ephocs from re-train
                EPOCHS = epochs_retrain

        ML.mean_absolute_error_last = 1000.0
        ML.mean_absolute_error_counter = 0
        ML.return_epochs = 0
        # only train if EPOCHS > 0
        if EPOCHS > 0:
            # a new optimal model using ModelCheckpoint
            history = model.fit(nparr_train_X, nparr_train_y,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_data=(nparr_test_X, nparr_test_y),
                                callbacks=[checkpointer, tensorboard, self.callbacks],
                                verbose=1)
            # uncomment for details
            #print(history)
            #model.summary()
            #model_png_path = path.join('temp', 'model') + '.png'
            #plot_model(model, to_file=model_png_path, show_shapes=True)
        # evaluate the model
        loss, mae = model.evaluate(nparr_test_X, nparr_test_y, verbose=0)
        # calculate the mean absolute error (inverse scaling)
        if scale:
            mean_absolute_error = scaler.inverse_transform([[mae]])[0][0]
        else:
            mean_absolute_error = mae

        print(f"{LOSS} loss:", loss)
        print("Mean Absolute Error:", mean_absolute_error)

        return loss, mean_absolute_error, self.return_epochs
