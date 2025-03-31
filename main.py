#!/usr/local/bin/python3
#
import downloadDataFin
import addFinancialParameters
import preProcessData
import trainLSTM
import testLSTM
import dataMerger
import buildChart

import pandas as pd

from os import path
from os import mkdir
from commonResources import *
from os import getcwd
from os import chdir
from os import remove



def process_ticker(ticker):
    """
    process ticker passed as parameter
    Params:
        ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
    Out:
        generate png-s for 1,3,6,12,24 months in web folder
    """

    # constants used to define the stage of the script
    STATUS_DOWNLOAD = 1             # download financial info from yahoo (downloadDataFin.py)
    STATUS_ADD_PARAMETERS = 2       # add financial parameters (addFinancialParameters.py)
    STATUS_TRAINING = 3             # train the network
    STATUS_MERGE_DATA = 4           # merge the data with old processed data
    STATUS_BUILD_CHART = 5          # build charts will past results
    STATUS_FINISHED = 6             # finished script
    # we start with download data
    int_status = STATUS_DOWNLOAD
    int_status = STATUS_ADD_PARAMETERS
    int_status = STATUS_TRAINING
    int_status = STATUS_MERGE_DATA
    int_status = STATUS_BUILD_CHART

    # this variable is used during development to perform only one step at a time
    single_step = True
    # single_step = False

    # constants used to define the forcast interval
    FORECAST_TEST = 0               # used for testing (when we optimize parameters)
    FORECAST_5_DAYS = 1             # used for 5 days forecast
    FORECAST_10_DAYS = 2            # used for 10 days forecast
    # we forecast for 5 days
    int_forecast = FORECAST_5_DAYS

    print('Start')


    # ticker : used for DOWNLOAD stage
    # interval for data download: used for DOWNLOAD stage
    interval = '1d'

    # features to use
    FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'MOM_1', 'MOM_3', 'MOM_5', 'MOM_10', 'MOM_20', 'ROC_2', 'ROC_5', 'ROC_10', 'ROC_20', 'ROC_40', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_40', 'STDEV_3', 'STDEV_5', 'STDEV_8',
                       'STDEV_15', 'dMOM_1', 'dMOM_3', 'dMOM_5', 'dMOM_10', 'dMOM_20', 'dROC_2', 'dROC_5', 'dROC_10', 'dROC_20', 'dROC_40', 'dEMA_5', 'dEMA_10', 'dEMA_20', 'dEMA_40', 'dSTDEV_3', 'dSTDEV_5', 'dSTDEV_8', 'dSTDEV_15']
    MINMAX_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_40', 'STDEV_3', 'STDEV_5', 'STDEV_8', 'STDEV_15']
    STANDARD_COLUMNS = ['MOM_1', 'MOM_3', 'MOM_5', 'MOM_10', 'MOM_20', 'ROC_2', 'ROC_5', 'ROC_10', 'ROC_20', 'ROC_40', 'dMOM_1', 'dMOM_3', 'dMOM_5', 'dMOM_10', 'dMOM_20', 'dROC_2', 'dROC_5', 'dROC_10', 'dROC_20', 'dROC_40',
                        'dEMA_5', 'dEMA_10', 'dEMA_20', 'dEMA_40', 'dSTDEV_3', 'dSTDEV_5', 'dSTDEV_8', 'dSTDEV_15']

    EPOCHS = [500]  # [100,200,300,400,500]
    if int_forecast == FORECAST_5_DAYS:
        NETWORK_LEN = [48]
        # prediction samples
        FUTURE_STEPS = [5]
        NEURONS = [256]
        N_LAYERS = [3]
        DROPOUT = [0.4]
    elif int_forecast == FORECAST_10_DAYS:
        NETWORK_LEN = [32, 48, 64]
        # prediction samples
        FUTURE_STEPS = [10]
        NEURONS = [256]
        N_LAYERS = [3]
        DROPOUT = [0.4]
    else: # FORECAST_TEST
        NETWORK_LEN = [48]  # [50] # [128, 256]   # [200,400]
        # prediction samples
        FUTURE_STEPS = [5]  # [3, 5, 10, 15]  # [48] # [64, 96] # [144,192] # [12,24,48]
        NEURONS = [256]  # [256]                   # 120  # 256
        N_LAYERS = [3]  # [3, 4, 5, 6]
        DROPOUT = [0.4]  # [0.3, 0.4, 0.6]


    BIDIRECTIONALS = [True] # [False, True] --> True is better
    SCALE = True
    TRAINING_LENGTH = 250       # one year

    # create TEMP folder if they don't exist
    if not path.isdir("temp"):
        # for temporary use
        mkdir("temp")

    # download data and check if new data is available
    bln_new_data_available = True
    if int_status == STATUS_DOWNLOAD:
        obj_download_data = downloadDataFin.DownloadFinancialData()
        obj_download_data.setup_folders()
        bln_new_data_available = obj_download_data.download_data(ticker=ticker, interval=interval)
        # if we don't have any new data there is nothing to compute
        if not bln_new_data_available:
            #  --> exit
            exit()
        # if is not 'single step' we move to the next stage
        if not single_step:
            int_status = STATUS_ADD_PARAMETERS

    # new data is available --> add financial data
    # Note: add financial also when training is expected --> because we need int_dataframe_length
    int_dataframe_length = 0
    if int_status == STATUS_ADD_PARAMETERS or int_status == STATUS_TRAINING:
        obj_fin_param = addFinancialParameters.FinancialParams()
        obj_fin_param.setup_folders()
        # when we are in ADD_PARAMETERS or we have new data we force an add_financial_data()
        if int_status == STATUS_ADD_PARAMETERS or bln_new_data_available:
            int_dataframe_length = obj_fin_param.add_financial_data(ticker=ticker, arr_LOOKUP_STEP=FUTURE_STEPS)
        else:
            # we only need the length
            int_dataframe_length = obj_fin_param.get_dataframe_length(ticker=ticker)
        if not single_step:
            int_status = STATUS_TRAINING

    if int_status == STATUS_TRAINING:

        int_offset = int_dataframe_length - 20      # process only 6 iterations
        #int_offset = int_dataframe_length - 50      # process only 36 iterations
        #int_offset = int_dataframe_length - 200
        #int_offset = int_dataframe_length - 300
        #int_offset = int_dataframe_length - 500
        #int_offset = int(int_dataframe_length / 2)       # 1940

        # minimum length is 14 so we force to 16+
        if int_offset < 16:
            int_offset = 16

        int_testing_length = int_dataframe_length - int_offset

        # default we process all data
        bln_process_only_new_data = False
        str_date_last = ''

        # create folders if they don't exist
        if not path.isdir(FOLDER_NAME_FOR_DATA_PREDICTIONS):
            # for data input (csv)
            mkdir(FOLDER_NAME_FOR_DATA_PREDICTIONS)

        for network in NETWORK_LEN:
            for neurons in NEURONS:
                for dropout in DROPOUT:
                    for bidirectional in BIDIRECTIONALS:
                        for n_layers in N_LAYERS:
                            for future_steps in FUTURE_STEPS:
                                str_file_name = f"{ticker}-seq-{network}-lookup-{future_steps}-layers-{n_layers}-units-{neurons}-dropout-{dropout}"
                                if bidirectional:
                                    str_file_name += "-b"
                                str_file_name = path.join(FOLDER_NAME_FOR_DATA_PREDICTIONS, f'{str_file_name}_pred.csv')
                                # check if file exist
                                if path.isfile(str_file_name):
                                    # record last date
                                    df_output_pred = pd.read_csv(str_file_name, sep=',')
                                    # getlast date to compare with available data
                                    str_date_last = df_output_pred['Date'].iloc[-1]
                                    bln_process_only_new_data = True
                                else:
                                    obj_file = open(str_file_name, 'w')
                                    obj_file.write('Date,Loss,MAE,Epochs,Prediction\n')
                                    obj_file.close()

                                # needed to know when for has reached last line from output_pred_
                                # NOTE: only used with bln_process_only_new_data
                                bln_matching_date = False
                                # we limit the end by exiting for with break (is prediction dependent)
                                for inta in range(int_testing_length + 1):

                                    old_epoch = 0
                                    for epoch in EPOCHS:
                                        new_epoch = epoch - old_epoch

                                        int_index = inta + int_offset

                                        # check if we have to exit for:
                                        # last run is:
                                        # training: 1 year
                                        # testing: network + future_steps(missing predictions)
                                        if int_index + 3 * future_steps >= int_dataframe_length + 1:
                                            # exit
                                            inta = int_dataframe_length
                                        else:
                                            # we add 3*future_steps to cover for last future_steps that is missing predictions (shifted back)
                                            TESING_LENGHT = 3 * future_steps
                                            # in the overlap process test set is increased by network
                                            obj_process_data = preProcessData.ProcessData()
                                            obj_process_data.setup_folders()

                                            if bln_process_only_new_data and not bln_matching_date:
                                                # this is the skip path
                                                # check if we have to process this index
                                                str_date_match = obj_process_data.get_date_at_index(ticker, index=int_index, overlap=network, training_length=TRAINING_LENGTH, testing_length=TESING_LENGHT)
                                                if str_date_last == str_date_match:
                                                    bln_matching_date = True
                                            else:
                                                # this is the training path
                                                obj_process_data.process_split_data_by_index(ticker, index=int_index, overlap=network, training_length=TRAINING_LENGTH, testing_length=TESING_LENGHT)

                                                obj_training = trainLSTM.ML()
                                                obj_training.setup_folders()
                                                obj_testing = testLSTM.ML()

                                                # NOTE: the testing set is limited inside process_split_data_by_index() to 2 * future_steps
                                                TESING_LENGHT = 2 * future_steps
                                                loss, mean_absolute_error, epochs_last = obj_training.train_model(ticker=ticker, epochs_start=new_epoch, epochs_retrain=0,
                                                                                                                 sequence_length=network, future_steps=future_steps, neurons=neurons, network_layers=n_layers,
                                                                                                                 drop_out=dropout, bidirectional=bidirectional, FEATURE_COLUMNS=FEATURE_COLUMNS,
                                                                                                                 scale=SCALE, MINMAX_COLUMNS=MINMAX_COLUMNS, STANDARD_COLUMNS=STANDARD_COLUMNS, testing_lenght=TESING_LENGHT,
                                                                                                                 exit_if_no_improvement_for=30, allow_model_loading=False)

                                                str_date_last, val_predicted = obj_testing.last_prediction_from_train_data(ticker=ticker, sequence_length=network, future_steps=future_steps, neurons=neurons, network_layers=n_layers,
                                                                                                                        drop_out=dropout, bidirectional=bidirectional, FEATURE_COLUMNS=FEATURE_COLUMNS, scale=SCALE)
                                                # save results
                                                obj_file = open(str_file_name, 'a')
                                                obj_file.write(str_date_last + ',' + str(loss) + ',' + str(mean_absolute_error) + ',' + str(epochs_last) + ',' + str(val_predicted) + '\n')
                                                obj_file.close()
                                                print('DateLast=' + str_date_last + ' , Predicted=' + str(val_predicted))
                                        old_epoch = epoch

        if not single_step:
            int_status = STATUS_MERGE_DATA

    if int_status == STATUS_MERGE_DATA:
        obj_data_merger = dataMerger.Merge()
        obj_data_merger.setup_folders()
        for network in NETWORK_LEN:
            for neurons in NEURONS:
                for dropout in DROPOUT:
                    for bidirectional in BIDIRECTIONALS:
                        for n_layers in N_LAYERS:
                            for future_steps in FUTURE_STEPS:
                                str_file_name = f"{ticker}-seq-{network}-lookup-{future_steps}-layers-{n_layers}-units-{neurons}-dropout-{dropout}"
                                if bidirectional:
                                    str_file_name += "-b"
                                # combine data
                                obj_data_merger.merge_data(ticker=ticker, future_steps=future_steps, str_file_name=str_file_name)

        #    for future_steps in FUTURE_STEPS:
        #        #strPathName = 'output_pred_' + str(future_steps) + '.csv'
        #        strFileName = path.join(FOLDER_NAME_FOR_DATA_PROCESSED, f'output_pred_{future_steps}.csv')
        #        objDataMerger.merge_data(ticker=ticker, future_steps=future_steps, input_path=strFileName)

        if not single_step:
            int_status = STATUS_BUILD_CHART

    if int_status == STATUS_BUILD_CHART:
        obj_chart = buildChart.Chart()
        obj_chart.setup_folders()
        last_close = 0
        pred_high = 0
        pred_low = 0

        for network in NETWORK_LEN:
            for neurons in NEURONS:
                for dropout in DROPOUT:
                    for bidirectional in BIDIRECTIONALS:
                        for n_layers in N_LAYERS:
                            for future_steps in FUTURE_STEPS:
                                str_file_name = f"{ticker}-seq-{network}-lookup-{future_steps}-layers-{n_layers}-units-{neurons}-dropout-{dropout}"
                                if bidirectional:
                                    str_file_name += "-b"
                                str_input_file_name = path.join(FOLDER_NAME_FOR_DATA_FINAL, f'{str_file_name}.csv')
                                # generate charts
                                last_close, pred_high, pred_low = obj_chart.generate_chart(ticker=ticker, future_steps=future_steps, bln_live=False, input_path=str_input_file_name, folder_name=str_file_name)

        # save to txt file (for web page)
        # ticker, last_close, pred_high, pred_low
        str_output_file_name = path.join(FOLDER_NAME_FOR_WEB, 'data.txt')
        obj_file = open(str_output_file_name, 'a')
        obj_file.write(f'{ticker},{last_close:.2f},{pred_high:.2f},{pred_low:.2f}\n')
        obj_file.close()

        if not single_step:
            int_status = STATUS_FINISHED


def main():

    # LIBIX - LifePath Index 2025 Account A - 0.05%
    # LINIX - LifePath Index 2030 Account A - 0.05%
    # LIJIX - LifePath Index 2035 Account A - 0.05%
    # LIKIX - LifePath Index 2040 Account A - 0.05%
    # LIHIX - LifePath Index 2045 Account A - 0.05%
    # LIPIX - LifePath Index 2050 Account A - 0.05%
    # LIVIX - LifePath Index 2055 Account A - 0.05%
    # LIZKX - LifePath Index 2060 Account A - 0.05%
    # LIWIX - LifePath Index 2065 Account A - 0.05%
    # LIRIX - LifePath Index Retirement Account A - 0.05%
    # ? - Stable Value Separate Account - 0.29%
    # VBTIX - Vanguard Total Bond Market Index Trust - 0.03%
    # ? - Eaton Vance Trust Co CIT High Yield Cl V - 0.39%
    # lSFIX - Loomis Sayles Core Plus Fixed Income D - 0.28%
    # VFFSX - Vanguard Institutional 500 Index Trust - 0.01%
    # ? - BHMS Large Cap Value Equity SMA - 0.25%
    # ? - Fidelity Contrafund Commingled Pool Cl 3 - 0.35%
    # WSMDX - William Blair Small-Mid Cap Growth SMA - 0.66%
    # ? -Vanguard Extended Market Index Trust - 0.04%
    # ? - AB US Small Mid-Cp Value CIT W Series P3 - 0.65%
    # BGITX - Baillie Gifford International Alpha CIT - 0.57%
    # VGTSX - Vanguard Total Intl Stock Index Trust - 0.06%
    # ? - BNY Mellon EB Global Real Estate Sec II - 0.54%

    tickers = ['VFFSX', 'VEXMX', 'AAPL']
    tickers = ['ADI']
    tickers = ['VOO']

    # check if destination folder exist
    strPathDestinationFile = path.join(FOLDER_NAME_FOR_WEB, 'data.txt')
    if path.isfile(strPathDestinationFile):
        # delete the file (we fill in the data, one row at a time in function process_ticker())
        remove(strPathDestinationFile)

    for ticker in tickers:
        print('************************************************************************************************************')
        print('********************************************  ' + ticker + '  *********************************************************')
        print('************************************************************************************************************')
        process_ticker(ticker=ticker)

    if False:
        str_path_destination_path = 'c:\\inetpub\\wwwroot\\'
        str_path_source_file = path.join(FOLDER_NAME_FOR_WEB,'data.txt')
        if path.isfile(str_path_source_file):
            # source file exist
            file_path = copy(str_path_source_file, str_path_destination_path)
            print(file_path)


if __name__ == "__main__":
    # this is required to fix some issues when script is called from script.
    print('current dir before: '+ getcwd())
    str_script_folder = path.dirname(path.realpath(__file__))
    str_script_folder_upper = path.split(str_script_folder)[0]
    # change current folder to file script folder
    chdir(str_script_folder_upper)
    # create folders if don't exist
    if not path.isdir("data"):
        # for data input (csv)
        mkdir("data")
    str_script_folder = path.join(str_script_folder_upper, 'data')
    chdir(str_script_folder)
    print('current dir after: ' + getcwd())
    # call main() function
    main()