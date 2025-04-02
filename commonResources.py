#!/usr/local/bin/python3

# definition of folder names:
FOLDER_NAME_FOR_DATA_IN = "1.data_in"   # where data is downloaded in csv file
FOLDER_NAME_FOR_DATA_FINANCIAL = "2.data_financial"   # where data is downloaded in csv file
FOLDER_NAME_FOR_DATA_PROCESSED = "3.data_proc"      # where training data and testing data is stored
FOLDER_NAME_FOR_DATA_PREDICTIONS = "4.data_pred"    # where predictions are stored
FOLDER_NAME_FOR_RESULTS = "4.results"    # where results (trained networks) are stored
FOLDER_NAME_FOR_LOGS = "4.logs"         # where log during training are stored
FOLDER_NAME_FOR_SCALER = "4.scaler"     # where scaler-s during training are stored
FOLDER_NAME_FOR_DATA_FINAL = "5.data_final" # where final merged data is stored
FOLDER_NAME_FOR_WEB = "6.web"           # where PNG data is stored

# constants used to define the forcast interval
FORECAST_TEST = 0  # used for testing (when we optimize parameters)
FORECAST_5_DAYS = 1  # used for 5 days forecast
FORECAST_10_DAYS = 2  # used for 10 days forecast

# constants used to define the stage of the script
STATUS_DOWNLOAD = 1  # download financial info from yahoo (downloadDataFin.py)
STATUS_ADD_PARAMETERS = 2  # add financial parameters (addFinancialParameters.py)
STATUS_TRAINING = 3  # train the network
STATUS_MERGE_DATA = 4  # merge the data with old processed data
STATUS_BUILD_CHART = 5  # build charts will past results
STATUS_FINISHED = 6  # finished script

# features to use
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'MOM_1', 'MOM_3', 'MOM_5', 'MOM_10', 'MOM_20', 'ROC_2', 'ROC_5', 'ROC_10', 'ROC_20', 'ROC_40', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_40', 'STDEV_3', 'STDEV_5', 'STDEV_8',
                   'STDEV_15', 'dMOM_1', 'dMOM_3', 'dMOM_5', 'dMOM_10', 'dMOM_20', 'dROC_2', 'dROC_5', 'dROC_10', 'dROC_20', 'dROC_40', 'dEMA_5', 'dEMA_10', 'dEMA_20', 'dEMA_40', 'dSTDEV_3', 'dSTDEV_5', 'dSTDEV_8', 'dSTDEV_15']
MINMAX_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_40', 'STDEV_3', 'STDEV_5', 'STDEV_8', 'STDEV_15']
STANDARD_COLUMNS = ['MOM_1', 'MOM_3', 'MOM_5', 'MOM_10', 'MOM_20', 'ROC_2', 'ROC_5', 'ROC_10', 'ROC_20', 'ROC_40', 'dMOM_1', 'dMOM_3', 'dMOM_5', 'dMOM_10', 'dMOM_20', 'dROC_2', 'dROC_5', 'dROC_10', 'dROC_20', 'dROC_40',
                    'dEMA_5', 'dEMA_10', 'dEMA_20', 'dEMA_40', 'dSTDEV_3', 'dSTDEV_5', 'dSTDEV_8', 'dSTDEV_15']