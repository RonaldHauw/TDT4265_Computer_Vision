import numpy as np
import time

import matplotlib
matplotlib.use('tkagg') #This line of code is to make matplotlib compatible with tensorflow
import matplotlib.pyplot as plt


class Save:

    def __init__(self):
        self.timestr = time.strftime("%Y%m%d-%H%M%S")

    def save_history_to_file(self,history,file_name):
        """
        :param history: history object from training either loss, acc, val_loss or val_acc
        :param file_name: name of the file where to save the history.
        :return: Nothing
        """
        numpy_history = np.array(history)
        np.savetxt(file_name, numpy_history, delimiter=",")

    def save_plots_raw_output(self,history):
        for label in list(history.history.keys()):
            file = "results/raw_output/" + label + "_" + self.timestr +".txt"
            self.save_history_to_file(history.history[label],file)

        print("Done saving raw output.")

        acc_plot_file_name = "results/plots/acc_plot" + self.timestr + ".png"
        loss_plot_file_name = "results/plots/loss_plot" + self.timestr + ".png"

        # summarize history for accuracy
        fig_acc = plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        fig_acc.savefig(acc_plot_file_name, dpi=fig_acc.dpi)

        # summarize history for loss
        fig_loss = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        fig_loss.savefig(loss_plot_file_name, dpi=fig_loss.dpi)

        print("Done saving plots.")
