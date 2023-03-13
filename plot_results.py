import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

Loss_COD_results_NPC = pd.read_csv("../Results/24022022_loss_COD_NPC.csv")
Loss_NH4_results_NPC = pd.read_csv('../Results/22022022_loss_NH4_NPC.csv')
Loss_PO4_results_NPC = pd.read_csv("../Results/23022022_loss_PO4_NPC.csv")
Loss_Biogas_results_NPC = pd.read_csv('../Results/24022022_loss_Biogas_NPC.csv')


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(Loss_NH4_results_NPC['Epoch'], Loss_NH4_results_NPC['loss_Bidirectional LSTM'], label='Training loss')
axs[0, 0].plot(Loss_NH4_results_NPC['Epoch'], Loss_NH4_results_NPC['val_loss_Bidirectional LSTM'], label='Validation loss')
axs[0, 0].set(xlabel='Epoch', ylabel='MSE')
axs[0, 0].legend(['Training loss', 'Validation loss'], loc='upper right')
axs[0, 0].set_title('Effluent NH4')
axs[0, 1].plot(Loss_PO4_results_NPC['Epoch'], Loss_PO4_results_NPC['loss_Bidirectional LSTM'], label='Training loss')
axs[0, 1].plot(Loss_PO4_results_NPC['Epoch'], Loss_PO4_results_NPC['val_loss_Bidirectional LSTM'], label='Validation loss')
axs[0, 1].set(xlabel='Epoch', ylabel='MSE')
axs[0, 1].legend(['Training loss', 'Validation loss'], loc='lower right')
axs[0, 1].set_title('Effluent PO4')
axs[1, 0].plot(Loss_COD_results_NPC['Epoch'], Loss_COD_results_NPC['loss_Bidirectional LSTM'], label='Training loss')
axs[1, 0].plot(Loss_COD_results_NPC['Epoch'], Loss_COD_results_NPC['val_loss_Bidirectional LSTM'], label='Validation loss')
axs[1, 0].set(xlabel='Epoch', ylabel='MSE')
axs[1, 0].legend(['Training loss', 'Validation loss'], loc='lower right')
axs[1, 0].set_title('Effluent COD')
axs[1, 1].plot(Loss_Biogas_results_NPC['Epoch'], Loss_Biogas_results_NPC['loss_Bidirectional LSTM'], label='Training loss')
axs[1, 1].plot(Loss_Biogas_results_NPC['Epoch'], Loss_Biogas_results_NPC['val_loss_Bidirectional LSTM'], label='Validation loss')
axs[1, 1].set(xlabel='Epoch', ylabel='MSE')
axs[1, 1].legend(['Training loss', 'Validation loss'], loc='lower right')
axs[1, 1].set_title('Biogas')
plt.savefig('../Results/Figures_Paper/Loss_figure.png')
plt.show()

'''fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(Loss_NH4_results_NPC['Time (d)'], Loss_NH4_results_NPC['NH4_meas_PCA (gNm3)'], label='Loss')
axs[0, 0].plot(Loss_NH4_results_NPC['Time (d)'], Loss_NH4_results_NPC['NH4_pred_PCA (gN/m3)'], label='Val. Loss')
axs[0, 0].set(xlabel='Epoch', ylabel='Loss')
axs[0, 0].legend(['Loss', 'Val. Loss'], loc='upper right')
axs[0, 0].set_title('A')
axs[0, 1].plot(PO4_results['Time (d)'], PO4_results['PO4_meas_PCA (gPm3)'], label='Meas.')
axs[0, 1].plot(PO4_results['Time (d)'], PO4_results['PO4_pred_PCA (gP/m3)'], label='Pred.')
axs[0, 1].set(xlabel='Time (d)', ylabel='PO4 (gP/m3)')
axs[0, 1].legend(['Meas.', 'Pred.'], loc='lower right')
axs[0, 1].set_title('B')
axs[1, 0].plot(Loss_COD_results_NPC['Epoch'], Loss_COD_results_NPC['loss_Bidirectional LSTM'], label='Loss')
axs[1, 0].plot(Loss_COD_results_NPC['Epoch'], Loss_COD_results_NPC['val_loss_Bidirectional LSTM'], label='Val. Loss')
axs[1, 0].set(xlabel='Time (d)', ylabel='COD (gCOD/m3)')
axs[1, 0].legend(['Meas.', 'Pred.'], loc='lower right')
axs[1, 0].set_title('C')
axs[1, 1].plot(Biogas_results['Time (d)'], Biogas_results['Biogas_meas_PCA (m3/d)'], label='Meas.')
axs[1, 1].plot(Biogas_results['Time (d)'], Biogas_results['Biogas_pred_PCA (m3/d)'], label='Pred.')
axs[1, 1].set(xlabel='Time (d)', ylabel='Biogas (m3/d)')
axs[1, 1].legend(['Meas.', 'Pred.'], loc='lower right')
axs[1, 1].set_title('D')
plt.savefig('../Data/testing_results.png')
plt.show()'''

'''for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')'''

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

'''# Plot test data vs prediction
def plot_future(self, prediction, model_name, y_test):
    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    time2 = self.raw_effluent[self.time_key][0::self.FREQUENCY]
    train_size = int(len(time2) * self.SPLIT_FRACTION)
    time = time2.iloc[train_size:][self.STEP_SIZE:]
    plt.plot(np.array(time), np.array(y_test), label='Test data')
    plt.plot(np.array(time), np.array(prediction), label='Prediction')

    plt.title('Test data vs prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Time (day)')
    plt.ylabel('Effluent NH4 (mgN/L)')
    plt.show()
# %%'''