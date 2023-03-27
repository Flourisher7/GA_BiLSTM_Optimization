# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
from scipy import stats
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras import backend as k
from math import floor

warnings.filterwarnings('ignore')
# % matplotlib inline

# %%
class BiLSTM:
    """This class encapsulates the neural networks models for time-series data types"""
    # Set Random Seed

    tf.random.set_seed(1234)  # random seed value for reproducible result
# %%
    # NN model hyperparameters
    STEP_SIZE = 1
    SPLIT_FRACTION = 0.8
    PAST = 240
    FUTURE = 24
    LEARNING_RATE = 0.0013572924182453008
    BATCH_SIZE = 27
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    HIDDEN_LAYERS = [27, 16]
    NUM_FEATURES = 3
    ACTIVATION = 'tanh'
    OPTIMIZER = 'ADAM'
    FREQUENCY = 1
# %%

    def __init__(self):
        """
        :param model: a neural network model such as LSTM
        """

        self.raw_process_data = pd.read_csv("Data/features.csv")
        self.raw_effluent = pd.read_csv('Data/targets.csv')

        # define parameters

        # Define a function to draw time_series plot
        # Raw Data Visualization
        self.titles = ["Inflow Total (ML/d)", "DIG 1 Raw Flow Rate (m3/d)", "RST Thicknd Sludge Volume",
                       "DAF12 TWAS Volume"]

        self.feature_keys3 = ['S_NH4 (gN/m3)', 'S_PO4 (gP/m3)', 'S_IC (gC/m3)',	'X_TSS (gSS/m3)', 'S_K (g/m3)',
                             'S_Mg (g/m3)',	'Q (m3/d)', 'Temp (oC)', 'S_Na (g/m3)', 'S_Cl (g/m3)', 'S_Ca (g/m3)',
                             'S_SO4 (g/m3)', 'COD (gCOD/m3)', 'CODeff (gCOD/m3)', 'S_NH4eff (gN/m3)', 'S_PO4eff (gP/m3)']
        self.feature_keys23 = ['S_NH4 (gN/m3)', 'S_PO4 (gP/m3)', 'S_IC (gC/m3)', 'X_TSS (gSS/m3)', 'S_K (g/m3)',
                             'S_Mg (g/m3)', 'Q (m3/d)', 'Temp (oC)', 'S_Na (g/m3)', 'S_Cl (g/m3)', 'S_Ca (g/m3)',
                             'S_SO4 (g/m3)', 'COD (gCOD/m3)', 'X_TSSdaf (gSS/m3)', 'X_TSSrst (gSS/m3)', 'Q_dig (m3/d)',
                             'Q_gas_norm (m3/d)']

        self.feature_keys33 = ['S_NH4eff (gN/m3)',]

        '''self.feature_keys = ['S_NH4 (gN/m3)', 'CODeff (gCOD/m3)', 'S_Ca (g/m3)', 'Temp (oC)', 'S_PO4eff (gP/m3)',
                              'X_TSS (gSS/m3)', 'Q_gas_norm (m3/d)', 'S_NH4eff (gN/m3)', 'S_PO4 (gP/m3)']'''


        '''self.feature_keys = ['S_NH4 (gN/m3)', 'S_IC (gC/m3)', 'S_K (g/m3)', 'S_Na (g/m3)', 'S_Cl (g/m3)',
                             'CODeff (gCOD/m3)', 'S_PO4eff (gP/m3)', 'Q_gas_norm (m3/d)']'''

        #  , 'Q_gas_norm (m3/d)'

        '''self.feature_keys = ['S_NH4 (gN/m3)', 'S_PO4 (gP/m3)', 'S_IC (gC/m3)', 'X_TSS (gSS/m3)', 'S_K (g/m3)',
                             'S_Mg (g/m3)', 'Q (m3/d)', 'Temp (oC)', 'S_Na (g/m3)', 'S_Cl (g/m3)', 'S_Ca (g/m3)',
                             'S_SO4 (g/m3)', 'COD (gCOD/m3)', 'COD_pco (gCOD/m3)', 'COD_pcu (gCOD/m3)',
                             'COD_reac16 (gCOD/m3)', 'X_TSSreac (gSS/m3)',  'S_NH4eff (gN/m3)', 'S_PO4eff (gP/m3)',
                             'Q_gas_norm (m3/d)']

        self.target_key = ['COD (gCOD/m3)']'''

        self.feature_keys = ['S_NH4 (gN/m3)', 'CODeff (gCOD/m3)', 'S_Ca (g/m3)', 'Temp (oC)']

        self.target_key = ['Q_gas_norm (m3/d)']

        self.colors = [
            'blue',
            'orange',
            'green',
            'red'
        ]

        self.time_key = "Time (d)"

        self.split_fraction = self.SPLIT_FRACTION
        # train_split = int(self.split_fraction * int(process_data.shape[0]))

    def scale(self, n):
        process_data = self.raw_process_data[self.feature_keys][0::n]
        target = self.raw_effluent[self.target_key][0::n]

        train_size = int(len(process_data) * self.split_fraction)
        train_process_data = process_data.iloc[:train_size]
        test_process_data = process_data.iloc[train_size:]
        train_target = target.iloc[:train_size]
        test_target = target.iloc[train_size:]

        # Data transformation
        # Features
        scaler_influent = MinMaxScaler().fit(train_process_data)
        train_scaled_influent = scaler_influent.transform(train_process_data)
        test_scaled_influent = scaler_influent.transform(test_process_data)
        # Targets
        scaler_effluent = MinMaxScaler().fit(train_target)
        train_scaled_effluent = scaler_effluent.transform(train_target)
        test_scaled_effluent = scaler_effluent.transform(test_target)

        X_train, y_train = self.create_dataset(train_scaled_influent,
                                               train_scaled_effluent, self.STEP_SIZE)
        X_test, y_test = self.create_dataset(test_scaled_influent,
                                             test_scaled_effluent, self.STEP_SIZE)

        return X_train, y_train, X_test, y_test, scaler_effluent

    def convert_params(self, params):
        # transform the layer sizes from float (possibly negative) values into hiddenLayerSizes tuples:
        if round(params[1]) <= 0:
            hidden_layer_sizes = [round(params[0])]
        elif round(params[2]) <= 0:
            hidden_layer_sizes = [round(params[0]), round(params[1])]
        elif round(params[3]) <= 0:
            hidden_layer_sizes = [round(params[0]), round(params[1]), round(params[2])]
        else:
            hidden_layer_sizes = [round(params[0]), round(params[1]), round(params[2]), round(params[3])]

        activation = ['tanh', 'relu', 'sigmoid'][floor(params[4])]
        optimizer = ['SDG', 'ADAM', 'Adagrad'][floor(params[5])]
        learning_rate = params[6]
        batch_size = floor(params[7])  # HAS TO BE AN INTEGER

        return hidden_layer_sizes, activation, optimizer, learning_rate, batch_size  #


    def get_mse(self, params):
        # parameters
        HIDDEN_LAYER_SIZES, ACTIVATION, OPTIMIZER, LEARNING_RATE, BATCH_SIZE = self.convert_params(params)  #

        X_train, y_train, X_test, y_test = self.scale(self.FREQUENCY)[0:4]
        #
        size = len(HIDDEN_LAYER_SIZES)
        model = Sequential()
        if size == 2:
            model.add(Bidirectional(LSTM(units=HIDDEN_LAYER_SIZES[size - 2], activation=ACTIVATION,
                                         return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Bidirectional(LSTM(units=HIDDEN_LAYER_SIZES[size - 1], activation=ACTIVATION,
                                         return_sequences=True)))
        elif size == 3:
            model.add(Bidirectional(LSTM(units=HIDDEN_LAYER_SIZES[size - 3], activation=ACTIVATION,
                                         return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Bidirectional(LSTM(units=HIDDEN_LAYER_SIZES[size - 2], activation=ACTIVATION,
                                         return_sequences=True)))
            model.add(Bidirectional(LSTM(units=HIDDEN_LAYER_SIZES[size - 1], activation=ACTIVATION)))
        elif size == 4:
            model.add(Bidirectional(LSTM(units=HIDDEN_LAYER_SIZES[size - 4], activation=ACTIVATION,
                                         return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Bidirectional(LSTM(units=HIDDEN_LAYER_SIZES[size - 3], activation=ACTIVATION,
                                         return_sequences=True)))
            model.add(Bidirectional(LSTM(units=HIDDEN_LAYER_SIZES[size - 2], activation=ACTIVATION,
                                         return_sequences=True)))
            model.add(Bidirectional(LSTM(units=HIDDEN_LAYER_SIZES[size - 1], activation=ACTIVATION)))

        model.add(Dense(1))
        # Compile model
        if OPTIMIZER == 'ADAM':
            opt = Adam(learning_rate=LEARNING_RATE)
        elif OPTIMIZER == 'SDG':
            opt = SGD(learning_rate=LEARNING_RATE)
        elif OPTIMIZER == 'Adagrad':
            opt =Adagrad(learning_rate=LEARNING_RATE)

        model.compile(optimizer=opt, loss='mse', metrics='mse')
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(X_train, y_train, epochs=self.EPOCHS,
                            validation_split=self.VALIDATION_SPLIT, batch_size=BATCH_SIZE,
                            shuffle=False, callbacks=[early_stop], verbose=0)

        mse = history.history['val_mse']
        #for val_mse in mse:
            # return mse
        return [np.abs(history.history['val_mse']).mean()]

    def format_params(self, params):
        hidden_layer_sizes, activation, optimizer, learning_rate, batch_size = self.convert_params(params)  # hidden_layer_sizes,
        return "'hidden_layer_sizes'={}\n "\
               "'activation'={}\n " \
               "'optimizer'='{}'\n " \
               "'learning_rate'={}\n " \
               "'batch_size'={} " \
            .format(hidden_layer_sizes, activation, optimizer, learning_rate, batch_size)  # "'hidden_layer_sizes'={}\n " \

    def show_raw_visualization(self, data):
        self.time_data = data[self.time_key]
        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
        )
        for i in range(len(self.feature_keys)):
            key = self.feature_keys[i]
            c = self.colors[i % (len(self.colors))]
            t_data = data[key]
            t_data.index = self.time_data
            t_data.head()
            ax = t_data.plot(
                ax=axes[i // 2, i % 2],
                color=c,
                title="{} - {}".format(self.titles[i], key),
                rot=25,
            )
            ax.legend([self.titles[i]])
        plt.tight_layout()

    # show_raw_visualization(self.influent)

# %%
    def show_heatmap(self, data):
        plt.matshow(data.corr())
        plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
        plt.gca().xaxis.tick_bottom()
        plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title("Feature Correlation Heatmap", fontsize=14)
        plt.show()

    #show_heatmap(influent[feature_keys])

# %%
    # Create Input
    def create_dataset(self, X, Y, step_size=1):
        Xs, ys = [], []

        for i in range(len(X) - step_size):
            v = X[i:i + step_size]
            Xs.append(v)
            ys.append(Y[i + step_size])

        return np.array(Xs), np.array(ys)

    # @staticmethod
    def coeff_determination(self, y_true, y_pred):
        SS_res = k.sum(k.square(y_true - y_pred))
        SS_tot = k.sum(k.square(y_true - k.mean(y_true)))
        return (1 - SS_res / (SS_tot + k.epsilon()))

    # 1 - Create BiLSTM model
    def create_bilstm(self, x_train):
        model = Sequential()
        # Input layer
        size = len(self.HIDDEN_LAYERS)
        if size == 2:
            model.add(Bidirectional(LSTM(units=self.HIDDEN_LAYERS[size - 2], activation=self.ACTIVATION,
                                         return_sequences=True), input_shape=(x_train.shape[1],
                                                                              x_train.shape[2])))
            model.add(Bidirectional(LSTM(units=self.HIDDEN_LAYERS[size - 1], activation=self.ACTIVATION)))
        elif size == 3:
            model.add(Bidirectional(LSTM(units=self.HIDDEN_LAYERS[size - 3], activation=self.ACTIVATION,
                                         return_sequences=True), input_shape=(x_train.shape[1],
                                                                              x_train.shape[2])))
            model.add(Bidirectional(LSTM(units=self.HIDDEN_LAYERS[size - 2], activation=self.ACTIVATION,
                                         return_sequences=True)))
            model.add(Bidirectional(LSTM(units=self.HIDDEN_LAYERS[size - 1], activation=self.ACTIVATION)))
        elif size == 4:
            model.add(Bidirectional(LSTM(units=self.HIDDEN_LAYERS[size - 4], activation=self.ACTIVATION,
                                         return_sequences=True), input_shape=(x_train.shape[1],
                                                                              x_train.shape[2])))
            model.add(Bidirectional(LSTM(units=self.HIDDEN_LAYERS[size - 3], activation=self.ACTIVATION,
                                         return_sequences=True)))
            model.add(Bidirectional(LSTM(units=self.HIDDEN_LAYERS[size - 2], activation=self.ACTIVATION,
                                         return_sequences=True)))
            model.add(Bidirectional(LSTM(units=self.HIDDEN_LAYERS[size - 1], activation=self.ACTIVATION)))

        model.add(Dense(1))
        # Compile model

        if self.OPTIMIZER == 'ADAM':
            opt = Adam(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'SDG':
            opt = SGD(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == 'Adagrad':
            opt = Adagrad(learning_rate=self.LEARNING_RATE)

        model.compile(optimizer=opt, loss='mse', metrics='mse')

        return model

    def create_vae(self, x_train):
        model = Sequential()

        return model
# %%
    # Fit the models
    # Takes the model and train it with training data for 100 epoch and batch_size = 16.
    # 20 % of training data is used for validata. shuffle = False gives better performance

    def fit_model(self, model, xtrain, ytrain):
        early_stop = keras.callbacks.EarlyStopping(monitor=
                                                   'val_loss', patience=10)
        history = model.fit(xtrain, ytrain, epochs=self.EPOCHS,
                            validation_split=self.VALIDATION_SPLIT, batch_size=self.BATCH_SIZE,
                            shuffle=False, callbacks=[early_stop], verbose=1)
        return history

    def get_metrics(self, history):
        mse = history.history['val_mse']
        print('history.history', history.history)
        for val_mse in mse:
            return val_mse  # np.abs(history.history['val_mse']).mean()
# %%

    # Evaluate models' performance
    # Plot train loss and validation loss

    def plot_loss(self, history, model_name):
        df_data1 = DataFrame(history.history['loss'], columns=['loss_' + model_name])
        df_data2 = DataFrame(history.history['val_loss'], columns=['val_loss_' + model_name])
        data_final = pd.concat([df_data1, df_data2], axis=1)
        data_final.to_csv('./Results_New/30072022_loss_NH4_All.csv', mode='a')  # change the file name accordingly
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Train vs Validation Loss for ' + model_name)
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['Train loss', 'Validation loss'], loc='upper right')
        plt.show()

    # %%
    # Compare prediction vs test data
    # Make prediction
    def prediction(self, model, x_test, scaler_effluent):
        prediction = model.predict(x_test)
        prediction = scaler_effluent.inverse_transform(prediction)

        return prediction

    # Plot test data vs prediction
    def plot_future(self, prediction, model_name, y_test):
        plt.figure(figsize=(10, 6))
        range_future = len(prediction)
        time2 = self.raw_effluent[self.time_key][0::self.FREQUENCY]
        train_size = int(len(time2) * self.SPLIT_FRACTION)
        time = time2.iloc[train_size:][self.STEP_SIZE:]
        df_data = DataFrame(np.array(time), columns=['Time (d)'])  # , 'Test data', 'Predicted data'])
        df_data1 = DataFrame(y_test, columns=['meas_'+self.target_key[0]])  # , 'Test data', 'Predicted data'])
        df_data2 = DataFrame(prediction, columns=['pred_'+self.target_key[0]])  # , 'Test data', 'Predicted data'])
        data_final = pd.concat([df_data, df_data1, df_data2], axis=1)
        data_final.to_csv('./Results_New/30072022_Prediction_NH4_All.csv')
        plt.plot(np.array(time), np.array(y_test), label='Test data')
        plt.plot(np.array(time), np.array(prediction), label='Prediction')
        # plt.plot(np.arange(range_future), np.array(y_test), label='Test data')
        # plt.plot(np.arange(range_future), np.array(prediction), label='Prediction')

        plt.title('Test data vs prediction for ' + model_name)
        plt.legend(loc='upper left')
        plt.xlabel('Time (day)')
        plt.ylabel(self.target_key[0])
        plt.savefig('../Results/PO4_Prediction_NH4_B.png')
        plt.show()
    # %%


    def evaluate_prediction(self, predictions, actual, model_name):
        errors = predictions - actual
        print(errors)
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        SS_res = np.sum(np.square(actual - predictions))
        SS_tot = np.sum(np.square(actual - np.mean(predictions)))
        coefficient_R2 = (1 - SS_res / SS_tot)

        print(model_name + ':')
        print('Mean Absolute Error: {:.4f}'.format(mae))
        print('Root Mean Square Error: {:.4f}'.format(rmse))
        print('Coefficient of Determination: {:.4f}'.format(coefficient_R2))

        print('')


    def model_evaluate(self, model,x_test, y_test):
        score, acc = model.evaluate(x_test, y_test)
        print('Test score:', score)
        print('Test accuracy:', acc)
        return acc
# %%

def main():

    bilstm_nn = BiLSTM()
    X_train, y_train, X_test, y_test, scaler_effluent = bilstm_nn.scale(bilstm_nn.FREQUENCY)


    # Print data shape
    print('X_train.shape: ', X_train.shape, '| y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape, '| y_test.shape: ', y_test.shape)

    #
    y_test2 = scaler_effluent.inverse_transform(y_test)
    y_train2 = scaler_effluent.inverse_transform(y_train)

    model_bilstm = bilstm_nn.create_bilstm(X_train)
    history_bilstm = bilstm_nn.fit_model(model_bilstm, X_train, y_train)
    bilstm_nn.plot_loss(history_bilstm, 'Bidirectional LSTM')
    prediction_bilstm = bilstm_nn.prediction(model_bilstm, X_test, scaler_effluent)
    bilstm_nn.plot_future(prediction_bilstm, 'Bidirectional LSTM', y_test2)
    bilstm_nn.evaluate_prediction(prediction_bilstm, y_test2, 'Bidirectional LSTM')
    print(bilstm_nn.get_metrics(history_bilstm))

    # NUM_FEATURES = len(bilstm_nn.feature_keys)

    # test = bilstm(12)

    # scores = []
    # calculate MSE for 'n' first features:
    # for n in range(1, len(test) + 1):
    # n_first_features = [1] * n + [0] * (len(test) - n)
    # score = test.get_mse(n_first_features)
    # print("%d first features: score = %f" % (n, score))
    # scores.append(score)


if __name__ == "__main__":
    main()
