#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd


from os import path
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from os import mkdir


class Chart:

    #
    def plot_png(self, ticker, objDataFrame, future_steps=5, intRangeX=1, folder_name=''):
        # plot it
    #    objFig, (objAxis0, objAxis1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})

        objFig = plt.figure()
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # the first subplot
        objAxis0 = plt.subplot(gs[0])

        strTitle = ticker + ' prediction for ' + str(future_steps) + ' days'
        objAxis0.set(title=strTitle)

        # plot Pred Low
        objAxis0.plot(objDataFrame['Date'], objDataFrame['PredLow'], label='Prediction Low', color='red', linestyle='dashed')
        # plot Pred High
        objAxis0.plot(objDataFrame['Date'], objDataFrame['PredHigh'], label='Prediction High', color='blue', linestyle='dashed')
        # plot Close
        objAxis0.plot(objDataFrame['Date'], objDataFrame['Close'], label='Close', color='green')

        # format xaxis
        plt.setp(objAxis0.get_xticklabels(), rotation=60)

        intdecimation = int(len(objDataFrame) / 30) + 1
        # Make a plot with major ticks that are multiples of 20 and minor ticks that
        # are multiples of 5.
        objAxis0.xaxis.set_major_locator(MultipleLocator(intdecimation))

        # For the minor ticks, use no labels; default NullFormatter.
        if intdecimation == 9:
            objAxis0.xaxis.set_minor_locator(MultipleLocator(3))

        # add grid
        objAxis0.grid(True)

        # bottom
        # the second subplot
        # shared axis X
        objAxis1 = plt.subplot(gs[1], sharex=objAxis0)
        # color is function of value
        my_cmap = plt.cm.get_cmap('RdYlGn')
        colors = my_cmap(objDataFrame['rMAE'])
        # plot rMAE
        objAxis1.bar(objDataFrame['Date'], objDataFrame['rMAE'], label='relative MAE', color=colors)
        # force scale 0 --> 1
        objAxis1.set_ylim(0,1)

        # format xaxis
        plt.setp(objAxis1.get_xticklabels(), rotation=60)
        # add grid
        objAxis1.grid(True)
        # Adjust the subplot layout
        plt.subplots_adjust(top=0.92, bottom=0.18, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=.0)

        # plt.show()
        strFolderName = path.join('web', folder_name)
        # create these folders if they does not exist
        if not path.isdir(strFolderName):
            mkdir(strFolderName)
        strFileName = path.join(strFolderName, f'{ticker}_{future_steps}_{intRangeX}.png')
        plt.savefig(strFileName, dpi=300)
        #strFileName = path.join('web', f'{ticker}_{future_steps}_{intRangeX}.csv')
        #objDataFrame.to_csv(strFileName, index=False)
        plt.close()


    def generate_chart(self, ticker, future_steps=5, blnLive=False, input_path='', folder_name=''):
        """
        Merge ticker with output_pred_X.csv.
        Params:
            ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
            future_steps (int): the prediction length
            input_path (str): the input file that need to be merged
        Out:
            chart from:
        """

    #    strFileName = path.join('data_proc', f'{ticker}_{future_steps}_final.csv')
        # load from CSVs
        objDataFrame = pd.read_csv(input_path, sep=',')

        intarrRangeX = [20, 60, 120, 250, 500]

        for intRangeX in intarrRangeX:
            objData = objDataFrame[['Date', 'Close', 'PredLow', 'PredHigh', 'rMAE']]
            objData = objData.tail(intRangeX)
            self.plot_png(ticker, objData, future_steps, intRangeX, folder_name)

        if blnLive:
            # Create figure and plot a stem plot with the date
            #fig, ax = plt.subplots(figsize=(8.8, 5), constrained_layout=True)
            objFig, objAxis = plt.subplots()
            strTitle = ticker + ' prediction for ' + str(future_steps) + ' days'
            objAxis.set(title=strTitle)

            # plot Pred Low
            plt.plot(objDataFrame['Date'], objDataFrame['PredLow'], label='Prediction Low', color='red', linestyle='dashed')
            # plot Pred High
            plt.plot(objDataFrame['Date'], objDataFrame['PredHigh'], label='Prediction High', color='blue', linestyle='dashed')
            # plot Close
            plt.plot(objDataFrame['Date'], objDataFrame['Close'], label='Close', color='green')

            # format xaxis
            plt.setp(objAxis.get_xticklabels(), rotation=60)

            plt.show()
            #strFileName = path.join('data', f'{ticker}_{future_steps}_full.png')
            #plt.savefig(strFileName, dpi=300)

        return objDataFrame['Close'].iloc[-1-future_steps], objDataFrame['PredLow'].iloc[-1], objDataFrame['PredHigh'].iloc[-1]
