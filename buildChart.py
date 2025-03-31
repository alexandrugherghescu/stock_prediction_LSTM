#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd


from os import path
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from os import mkdir
from commonResources import *


class Chart:


    def setup_folders(self):
        """
        Setup folder data_in for download (if not already available)
        Params:
            None
        Out:
            None
        """
        # create folders if they don't exist
        if not path.isdir(FOLDER_NAME_FOR_WEB):
            # for data input (csv)
            mkdir(FOLDER_NAME_FOR_WEB)


    def plot_png(self, ticker, obj_data_frame, future_steps=5, int_rangeX=1, folder_name=''):
        # plot it
    #    objFig, (obj_axis0, obj_axis1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})

        objFig = plt.figure()
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # the first subplot
        obj_axis0 = plt.subplot(gs[0])

        str_title = ticker + ' prediction for ' + str(future_steps) + ' days'
        obj_axis0.set(title=str_title)

        # plot Pred Low
        obj_axis0.plot(obj_data_frame['Date'], obj_data_frame['PredLow'], label='Prediction Low', color='red', linestyle='dashed')
        # plot Pred High
        obj_axis0.plot(obj_data_frame['Date'], obj_data_frame['PredHigh'], label='Prediction High', color='blue', linestyle='dashed')
        # plot Close
        obj_axis0.plot(obj_data_frame['Date'], obj_data_frame['Close'], label='Close', color='green')

        # format xaxis
        plt.setp(obj_axis0.get_xticklabels(), rotation=60)

        int_decimation = int(len(obj_data_frame) / 30) + 1
        # Make a plot with major ticks that are multiples of 20 and minor ticks that
        # are multiples of 5.
        obj_axis0.xaxis.set_major_locator(MultipleLocator(int_decimation))

        # For the minor ticks, use no labels; default NullFormatter.
        if int_decimation == 9:
            obj_axis0.xaxis.set_minor_locator(MultipleLocator(3))

        # add grid
        obj_axis0.grid(True)

        # bottom
        # the second subplot
        # shared axis X
        obj_axis1 = plt.subplot(gs[1], sharex=obj_axis0)
        # color is function of value
        my_cmap = plt.cm.get_cmap('RdYlGn')
        colors = my_cmap(obj_data_frame['rMAE'])
        # plot rMAE
        obj_axis1.bar(obj_data_frame['Date'], obj_data_frame['rMAE'], label='relative MAE', color=colors)
        # force scale 0 --> 1
        obj_axis1.set_ylim(0,1)

        # format xaxis
        plt.setp(obj_axis1.get_xticklabels(), rotation=60)
        # add grid
        obj_axis1.grid(True)
        # Adjust the subplot layout
        plt.subplots_adjust(top=0.92, bottom=0.18, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=.0)

        # plt.show()
        str_folder_name = path.join(FOLDER_NAME_FOR_WEB, folder_name)
        # create these folders if they does not exist
        if not path.isdir(str_folder_name):
            mkdir(str_folder_name)
        str_file_name = path.join(str_folder_name, f'{ticker}_{future_steps}_{int_rangeX}.png')
        plt.savefig(str_file_name, dpi=300)
        #str_file_name = path.join(FOLDER_NAME_FOR_WEB, f'{ticker}_{future_steps}_{int_rangeX}.csv')
        #obj_data_frame.to_csv(str_file_name, index=False)
        plt.close()


    def generate_chart(self, ticker, future_steps=5, bln_live=False, input_path='', folder_name=''):
        """
        Merge ticker with output_pred_X.csv.
        Params:
            ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
            future_steps (int): the prediction length
            input_path (str): the input file that need to be merged
            folder_name (str): the input folder where png-s will be generated
        Out:
            chart from:
        """

    #    strFileName = path.join('3.data_proc', f'{ticker}_{future_steps}_final.csv')
        # load from CSVs
        obj_data_frame = pd.read_csv(input_path, sep=',')

        int_arr_rangeX = [20, 60, 120, 250, 500]

        for int_rangeX in int_arr_rangeX:
            obj_data = obj_data_frame[['Date', 'Close', 'PredLow', 'PredHigh', 'rMAE']]
            obj_data = obj_data.tail(int_rangeX)
            self.plot_png(ticker, obj_data, future_steps, int_rangeX, folder_name)

        if bln_live:
            # Create figure and plot a stem plot with the date
            #fig, ax = plt.subplots(figsize=(8.8, 5), constrained_layout=True)
            obj_fig, obj_axis = plt.subplots()
            str_title = ticker + ' prediction for ' + str(future_steps) + ' days'
            obj_axis.set(title=str_title)

            # plot Pred Low
            plt.plot(obj_data_frame['Date'], obj_data_frame['PredLow'], label='Prediction Low', color='red', linestyle='dashed')
            # plot Pred High
            plt.plot(obj_data_frame['Date'], obj_data_frame['PredHigh'], label='Prediction High', color='blue', linestyle='dashed')
            # plot Close
            plt.plot(obj_data_frame['Date'], obj_data_frame['Close'], label='Close', color='green')

            # format xaxis
            plt.setp(obj_axis.get_xticklabels(), rotation=60)

            plt.show()
            #strFileName = path.join('data', f'{ticker}_{future_steps}_full.png')
            #plt.savefig(strFileName, dpi=300)

        return obj_data_frame['Close'].iloc[-1-future_steps], obj_data_frame['PredLow'].iloc[-1], obj_data_frame['PredHigh'].iloc[-1]
