# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
from scipy import stats
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
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
    STEP_SIZE = 24
    SPLIT_FRACTION = 0.5
    PAST = 240
    FUTURE = 24
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    UNITS = 64
    NUM_FEATURES = 3
    ACTIVATION = 'relu'

# %%

    def __init__(self):
        """
        :param model: a neural network model such as LSTM
        """

        # read and load CSV file in a DataFrame
        self.raw_process_data = pd.read_csv("../Data/influent2.csv")
        self.raw_effluent = pd.read_csv('../Data/effluent2.csv')

        # self.raw_process_data = pd.read_csv('../Data/influent3.csv')
        # self.raw_effluent = pd.read_csv('../Data/digesterout2.csv')

        # define parameters

        # Define a function to draw time_series plot
        # Raw Data Visualization
        self.titles = ["Inflow Total (ML/d)", "DIG 1 Raw Flow Rate (m3/d)", "RST Thicknd Sludge Volume",
                       "DAF12 TWAS Volume"]

        self.feature_keys2 = ['S_NH4 (gN/m3)', 'S_PO4 (gP/m3)', 'S_IC (gC/m3)', 'X_TSS (gSS/m3)', 'S_K (g/m3)',
                             'S_Mg (g/m3)', 'Q (m3/d)', 'Temp (oC)', 'S_Na (g/m3)', 'S_Cl (g/m3)', 'S_Ca (g/m3)',
                             'S_SO4 (g/m3)', 'COD (gCOD/m3)', 'S_NH4eff (gN/m3)', 'S_PO4eff (gP/m3)']
        self.feature_keys3 = ['S_NH4 (gN/m3)', 'S_PO4 (gP/m3)', 'S_IC (gC/m3)', 'X_TSS (gSS/m3)', 'S_K (g/m3)',
                             'S_Mg (g/m3)', 'Q (m3/d)', 'Temp (oC)', 'S_Na (g/m3)', 'S_Cl (g/m3)', 'S_Ca (g/m3)',
                             'S_SO4 (g/m3)', 'COD (gCOD/m3)', 'X_TSSdaf (gSS/m3)', 'X_TSSrst (gSS/m3)', 'Q_dig (m3/d)',
                             'Q_gas_norm (m3/d)']
        # Best Ever Individual =  [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        # COD [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

        self.feature_keys = ['S_NH4 (gN/m3)', 'S_IC (gC/m3)', 'Q (m3/d)', 'COD (gCOD/m3)', 'S_NH4eff (gN/m3)']

        self.target_key = ['S_NH4 (gN/m3)']

        self.colors = [
            'blue',
            'orange',
            'green',
            'red'
        ]

        self.time_key = "Time (d)"

        self.split_fraction = self.SPLIT_FRACTION
        # train_split = int(self.split_fraction * int(process_data.shape[0]))

        '''self.process_data = self.raw_process_data[self.feature_keys]
        self.target = self.raw_effluent[self.target_key]

        self.train_split = int(self.split_fraction * int(self.process_data.shape[0]))
        self.train_size = int(len(self.process_data) * self.split_fraction)
        self.train_process_data = self.process_data.iloc[:self.train_size]
        self.test_process_data = self.process_data.iloc[self.train_size:]
        self.train_target = self.target.iloc[:self.train_size]
        self.test_target = self.target.iloc[self.train_size:]

        # Data transformation
        # Features
        self.scaler_influent = MinMaxScaler().fit(self.train_process_data)
        self.train_scaled_influent = self.scaler_influent.transform(self.train_process_data)
        self.test_scaled_influent = self.scaler_influent.transform(self.test_process_data)
        # Targets
        self.scaler_effluent = MinMaxScaler().fit(self.train_target)
        self.train_scaled_effluent = self.scaler_effluent.transform(self.train_target)
        self.test_scaled_effluent = self.scaler_effluent.transform(self.test_target)'''
        #

        '''self.X_train, self.y_train = self.create_dataset(self.train_scaled_influent,
                                                         self.train_scaled_effluent, self.STEP_SIZE)
        self.X_test, self.y_test = self.create_dataset(self.test_scaled_influent,
                                                       self.test_scaled_effluent, self.STEP_SIZE)

        self.y_test2 = self.scaler_effluent.inverse_transform(self.y_test)
        self.y_train2 = self.scaler_effluent.inverse_transform(self.y_train)
'''
    def scale(self, n):
        process_data = self.raw_process_data[self.feature_keys][0::n]
        target = self.raw_effluent[self.target_key]

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
        frequency = floor(params[0])
        return frequency  #


    def get_mse(self, params):
        # parameters
        FREQUENCY = self.convert_params(params)  #

        X_train, y_train, X_test, y_test = self.scale(FREQUENCY)[0:4]
        #
        model = Sequential()
        # Input layer

        model.add(Bidirectional(LSTM(units=self.UNITS, activation=self.ACTIVATION, return_sequences=True),
                                input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Bidirectional(LSTM(units=self.UNITS, activation=self.ACTIVATION, return_sequences=True)))

        model.add(Dense(1))
        # Compile model
        opt = Adam(learning_rate=self.LEARNING_RATE)

        model.compile(optimizer=opt, loss='mse', metrics='mse')
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(X_train, y_train, epochs=self.EPOCHS,
                            validation_split=self.VALIDATION_SPLIT, batch_size=self.BATCH_SIZE,
                            shuffle=False, callbacks=[early_stop], verbose=0)

        mse = history.history['val_mse']
        #for val_mse in mse:
            # return mse
        return [np.abs(history.history['val_mse']).mean()]

    def format_params(self, params):
        frequency = self.convert_params(params)  # hidden_layer_sizes,
        return "'frequency'='{}'"\
            .format(frequency)  # "'hidden_layer_sizes'={}\n " \

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

    # 1 - Create BiLSTM model
    def create_bilstm(self, x_train):
        model = Sequential()
        # Input layer
        model.add(Bidirectional(LSTM(units=self.UNITS, return_sequences=True),
                                input_shape=(x_train.shape[1], x_train.shape[2])))
        # Hidden layer
        model.add(Bidirectional(LSTM(units=self.UNITS)))
        model.add(Dense(1))
        # Compile model

        opt = Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss='mse', metrics='mse')

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
        for val_mse in mse:
            return val_mse  # np.abs(history.history['val_mse']).mean()
# %%

    # Evaluate models' performance
    # Plot train loss and validation loss

    def plot_loss(self, history, model_name):
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
        plt.plot(np.arange(range_future), np.array(y_test),
                 label='Test data')
        plt.plot(np.arange(range_future),
                 np.array(prediction), label='Prediction')

        plt.title('Test data vs predition for ' + model_name)
        plt.legend(loc='upper left')
        plt.xlabel('Time (day)')
        plt.ylabel('Effluent NH4 (mgN/L)')
        plt.show()
    # %%

    def evaluate_prediction(self, predictions, actual, model_name):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()

        print(model_name + ':')
        print('Mean Absolute Error: {:.4f}'.format(mae))
        print('Root Mean Square Error: {:.4f}'.format(rmse))

        print('')


    def model_evaluate(self, model,x_test, y_test):
        score, acc = model.evaluate(x_test, y_test)
        print('Test score:', score)
        print('Test accuracy:', acc)
        return acc
# %%

def main():

    bilstm_nn = BiLSTM()
    X_train, y_train, X_test, y_test, scaler_effluent = bilstm_nn.scale(1)


    # Print data shape
    #print('X_train.shape: ', X_train.shape, '| y_train.shape: ', y_train.shape)
    #print('X_test.shape: ', X_test.shape, '| y_test.shape: ', y_test.shape)

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


    #NUM_FEATURES = len(bilstm_nn.feature_keys)

    # test = bilstm(12)

    # scores = []
    # calculate MSE for 'n' first features:
    # for n in range(1, len(test) + 1):
        # n_first_features = [1] * n + [0] * (len(test) - n)
        # score = test.get_mse(n_first_features)
        # print("%d first features: score = %f" % (n, score))
        # scores.append(score)

'''    params = [11.394267984578837, -4.624838671659996, -4.499413632617615, -13.303677855535318, 2.2086771712778734, 2.029421762781311, 0.08922873881371406, 21.304082489441242]
    params2 = [[11.394267984578837, -4.624838671659996, -4.499413632617615, -13.303677855535318, 2.2086771712778734,
      2.029421762781311, 0.08922873881371406, 21.304082489441242],
     [9.219218196852704, -4.553041708428945, -5.627240503927933, -4.839341356899128, 0.07958137308190702,
      0.5963141144092589, 0.06502345533417438, 28.17412220904825],
     [7.204406220406967, 3.8389852581386315, 6.1886091335565325, -19.80503720965817, 2.416651936246591,
      2.0937200455696923, 0.034091026600147394, 22.332192497176724],
     [14.572130722067811, 0.04891817668940135, -8.145083132397042, -17.09850869499608, 2.541635604676032,
      1.8105743680693065, 0.08073211450011059, 30.945976800407266],
     [10.362280914547007, 9.59673645969056, -2.42931245583293, -3.4387810618031907, 2.4873845880947316,
      1.854940737340374, 0.08618451934104665, 28.660282178851432],
     [12.045718362149234, -4.312634245165067, -5.442034486969063, -11.318361091936785, 0.23929613879395886,
      0.6981398681967296, 0.010190042798031939, 24.169604046651383],
     [11.356844442644002, 0.47248268455126397, -2.5963806576623476, -13.71478907685537, 0.800666488325291,
      2.8090271085497696, 0.0648387349861347, 29.136965085004825],
     [6.71138648198097, 5.9369019692552385, -6.731950124761433, -8.616336747270566, 2.9675805285591492,
      1.9193592798024246, 0.055739279403087164, 30.269213764848118],
     [13.428519201898096, 6.639998673193672, -5.419038560717913, -19.036992682878868, 0.9460436911291867,
      0.8029548870511325, 0.021177186074274015, 34.14364571502581],
     [13.763676264726689, -0.2798317880228316, 3.1087733058976, -8.131042968180072, 2.7427282216318902,
      1.376096705909609, 0.026561528633155442, 23.699412615409752],
     [10.613681341631509, -1.058875872155971, 1.6917198044708108, 6.934686508074311, 1.1978021149160514,
      0.6577429567126928, 0.09975400688886152, 27.64289440514697],
     [5.9090941217379385, -4.293254368628982, -7.8070173929868165, -1.1766187489072983, 2.3754460137245292,
      1.2660577404322524, 0.006446417844580518, 25.72428929759805],
     [14.961213802400968, 2.9367151764870556, 9.421567552272364, 5.82339106703494, 0.03443158480651609,
      2.161444736261224, 0.06820286586575483, 28.054554956131927],
     [7.668251899525428, 4.6144269786971215, -7.768956528082471, -6.95704247992685, 1.36071739528129,
      2.8604939666357194, 0.08759770874378159, 23.95083576126636],
     [10.005861130502982, -2.3202217920480295, 8.25255678689641, 6.115557095103007, 0.8950359295544501,
      1.9162095351031496, 0.06093612412267342, 22.29258902824452],
     [12.625108000751514, 3.0906854517943856, 5.572529572611165, -4.089389834144676, 0.0017151164877025724,
      0.9721440149570146, 0.002045726564344647, 33.93647924396926],
     [13.787218778231843, 7.474982940417691, -3.8497174919467714, -18.262245005174375, 2.6331507880129177,
      2.8399013864486844, 0.00865677986158109, 27.289856949749208],
     [5.692125184683836, 6.409032478858473, 5.316688586139755, -16.148256065007118, 1.4253718519180953,
      1.6488609768913367, 0.026579157231111903, 33.08649561627886],
     [9.23137940200887, -1.823026918368769, 0.7859217755891663, 1.8979320726992839, 0.603252039105698,
      0.934837157611384, 0.09951542073042338, 29.748170864591803],
     [9.381000839145042, 2.7636376155338596, -7.579916082634686, -13.259079889053279, 1.0139186008802186,
      1.7643378466532427, 0.02308846178639804, 23.30326076677339]]'''

    # param = bilstm_nn.convert_params(params2)

    # print(param)
    # print(type(params[2]))
    # print(params2[2])
    # print("converted", bilstm_nn.convert_params(params2[2]))
    # mse = bilstm_nn.get_mse(params2[2])
    # print("Mean Squared Error: params", 2, "is", mse)

    # for i in range(len(params2)):
        # print(i)
        # print(params2[i])
        # mse = bilstm_nn.get_mse(params2[i])
        # print("Mean Squared Error: ", i, "is", mse)

    # print(type(mse))

    # form = bilstm_nn.format_params(params)
    # print(form)

if __name__ == "__main__":
    main()