import pandas as pd
import numpy as np
import math
import keras
from scipy.stats import norm
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras.optimizers as opt
import time
import sys


# Given Data
no_of_paths = int(sys.argv[1])
no_of_hidden_nodes = int(sys.argv[2])
# value = float(sys.argv[3])
value = 1

w_dict = {}
no_of_output_nodes = 1
weights_dict = {}
no_of_assets = 5

# cor_mat = [[1.0, 0.79, 0.82, 0.91, 0.84],
#            [0.79, 1.0, 0.73, 0.80, 0.76],
#            [0.82, 0.73, 1.0, 0.77, 0.72],
#            [0.91, 0.80, 0.77, 1.0, 0.90],
#            [0.84, 0.76, 0.72, 0.90, 1.0]]

cor_mat = [[1.0, 1.0, 1.0, 1.0, 1.0],
           [1.0, 1.0, 1.0, 1.0, 1.0],
           [1.0, 1.0, 1.0, 1.0, 1.0],
           [1.0, 1.0, 1.0, 1.0, 1.0],
           [1.0, 1.0, 1.0, 1.0, 1.0]]

vol_list = np.array([0.518, 0.648, 0.623, 0.570, 0.530])
Notional = 100
curr_stock_price = np.ones(no_of_assets) * value
t = 1
k = 1
no_of_exercise_days = 1
r = 0.05
w = np.array([0.381, 0.065, 0.057, 0.270, 0.227])
w = w.reshape(-1, 1)
exercise_days = np.array([float(i / no_of_exercise_days) for i in range(1, no_of_exercise_days + 1)])
dt = t / no_of_exercise_days


def generate_covariance_from_correlation(cor_mat, vol_list, dt):
    vol_diag_mat = np.diag(vol_list)
    cov_mat = np.dot(np.dot(vol_diag_mat, cor_mat), vol_diag_mat) * dt
    return cov_mat


def multi_variate_gbm_simulation(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets,
                                 curr_stock_price, r, vol_list, cov_mat, t):
    zero_mean = np.zeros(no_of_assets)

    dw_mat = np.random.multivariate_normal(zero_mean, cov_mat, (no_of_paths, no_of_exercise_days))
    dt = t / no_of_exercise_days

    sim_ln_stock_mat = np.zeros((no_of_paths, no_of_exercise_days + 1, no_of_assets))
    sim_ln_stock_mat[:, 0] = np.tile(np.log(curr_stock_price), (no_of_paths, 1))
    base_drift = np.tile((np.add(np.full(no_of_assets, r), - 0.5 * np.square(vol_list))), (no_of_paths, 1)) * dt

    for day in range(1, no_of_exercise_days + 1):
        curr_drift = sim_ln_stock_mat[:, day - 1] + base_drift
        sim_ln_stock_mat[:, day] = curr_drift + dw_mat[:, day - 1]

    sim_stock_mat = np.exp(sim_ln_stock_mat)
    return sim_stock_mat


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


# Generate covariance matrix without considering dt
cov_mat = generate_covariance_from_correlation(cor_mat, vol_list, dt)


def pricer_arithmetic_pre(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets, w, sim_stock_mat,
                          batch_size, no_of_epochs, no_of_hidden_nodes, no_of_output_nodes, t, model):

    continuation_value = np.zeros((no_of_paths, 1))
    stock_vec = sim_stock_mat[:, no_of_exercise_days]
    intrinsic_value = np.maximum(k - np.dot(stock_vec, w), 0)
    continuation_value = intrinsic_value
    
    # Finding intrinsic value of the option for all paths and exercise days
    for day in range(no_of_exercise_days - 1, no_of_exercise_days - 2, -1):
        stock_vec = sim_stock_mat[:, day + 1]
        option_value = continuation_value

        X_train = np.log(stock_vec)
        X_train = X_train.reshape(-1, 5)
        Y_train = option_value
        Y_train = np.asarray(Y_train)
        Y_train.reshape(-1, 1, 1)

        nnet_output = model.fit(X_train, Y_train, epochs=no_of_epochs[day], batch_size=batch_size, verbose=0,
                                validation_split=0.2, callbacks=[es])

        w_vect = np.array(nnet_model.layers[0].get_weights()[0])
        w_vect_2 = np.array(nnet_model.layers[1].get_weights()[0])
        strikes = np.array(nnet_model.layers[0].get_weights()[1])
        bias_2 = np.array(nnet_model.layers[1].get_weights()[1])
        strikes = np.asarray(strikes)

        stock_vec = sim_stock_mat[:, day]
        x = np.log(stock_vec) + np.tile(((r - 0.5 * np.square(vol_list)) * dt).reshape(1, no_of_assets),
                                        (no_of_paths, 1))
        opt_val = np.zeros((no_of_paths, 1))

        for node in range(0, no_of_hidden_nodes):
            w_o = w_vect[:, node]
            w_o = w_o.reshape(no_of_assets, 1)
            mu = np.dot(x, w_o) + strikes[node]
            var = np.dot(np.dot(w_o.T, cov_mat), w_o)
            sd = var ** 0.5
            ft = mu * (1 - norm(0, sd).cdf(-mu))
            st = (sd / (2 * math.pi) ** 0.5) * np.exp(-0.5 * (mu / sd) ** 2)
            opt_val = opt_val + w_vect_2[node] * (ft + st)

        continuation_value = (opt_val + bias_2) * np.exp(-r * dt)

    return model



def pricer_bermudan_options_by_nn(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets, w, sim_stock_mat,
                                  batch_size, no_of_epochs, no_of_hidden_nodes, no_of_output_nodes, t, model):
    # Creating zero n-d arrays for intrinsic value, continuation value and option value
    continuation_value = np.zeros((no_of_paths, 1))
    stock_vec = sim_stock_mat[:, no_of_exercise_days]
    intrinsic_value = np.maximum(k - np.dot(stock_vec, w), 0)
    continuation_value = intrinsic_value
    
    for day in range(no_of_exercise_days - 1, -1, -1):
        stock_vec = sim_stock_mat[:, day + 1]
        

        option_value = continuation_value

        X_train = np.log(stock_vec)
        X_train = X_train.reshape(-1, 5)
        Y_train = option_value
        Y_train = np.asarray(Y_train)
        Y_train.reshape(-1, 1, 1)

        nnet_output = model.fit(X_train, Y_train, epochs=no_of_epochs[day], batch_size=batch_size, verbose=1,
                                validation_split=0.2, callbacks=[es])

        w1 = nnet_model.layers[0].get_weights()
        w2 = nnet_model.layers[1].get_weights()
        w_dict[day] = ([w1, w2])
        w_vect = np.array(nnet_model.layers[0].get_weights()[0])
        w_vect_2 = np.array(nnet_model.layers[1].get_weights()[0])
        strikes = np.array(nnet_model.layers[0].get_weights()[1])
        bias_2 = np.array(nnet_model.layers[1].get_weights()[1])
        strikes = np.asarray(strikes)
        weights_dict[day] = ([nnet_model.layers[0].get_weights()[0], nnet_model.layers[1].get_weights()[0],
                              nnet_model.layers[0].get_weights()[1], nnet_model.layers[1].get_weights()[1]])
        stock_vec = sim_stock_mat[:, day]
        x = np.log(stock_vec) + np.tile(((r - 0.5 * np.square(vol_list)) * dt).reshape(1, no_of_assets),
                                        (no_of_paths, 1))
        opt_val = np.zeros((no_of_paths, 1))

        for node in range(0, no_of_hidden_nodes):
            w_o = w_vect[:, node]
            w_o = w_o.reshape(no_of_assets, 1)
            mu = np.dot(x, w_o) + strikes[node]
            var = np.dot(np.dot(w_o.T, cov_mat), w_o)
            sd = var ** 0.5
            ft = mu * (1 - norm(0, sd).cdf(-mu))
            st = (sd / (2 * math.pi) ** 0.5) * np.exp(-0.5 * (mu / sd) ** 2)
            opt_val = opt_val + w_vect_2[node] * (ft + st)

        continuation_value = (opt_val + bias_2) * np.exp(-r * dt)

    return (np.mean(continuation_value))

price_list = []
for i in range(0, 30):
    
    sim_stock_mat = multi_variate_gbm_simulation(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets,
                                                 curr_stock_price, r, vol_list, cov_mat, t)
    batch_size = int(no_of_paths / 10)
    no_of_epochs = np.array([100, 100, 100, 500, 500, 500, 500, 2000])

    nnet_model = Sequential()
    nnet_model.add(Dense(no_of_hidden_nodes, activation='relu', kernel_initializer='random_uniform'))
    nnet_model.add(Dense(1, activation='linear', kernel_initializer='normal'))
    nnet_model.compile(optimizer=opt.Adam(lr=0.001), loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5)

    nnet_model = pricer_arithmetic_pre(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets, w, sim_stock_mat,
                                       batch_size, no_of_epochs, no_of_hidden_nodes, no_of_output_nodes, t, nnet_model)
    K.set_value(nnet_model.optimizer.lr, 5e-4)

    sim_stock_mat = multi_variate_gbm_simulation(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets,
                                                 curr_stock_price, r, vol_list, cov_mat, t)

    nnet_model = pricer_arithmetic_pre(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets, w, sim_stock_mat,
                                      batch_size, no_of_epochs, no_of_hidden_nodes, no_of_output_nodes, t, nnet_model)

    K.set_value(nnet_model.optimizer.lr, 1e-3)

    price = pricer_bermudan_options_by_nn(no_of_paths, no_of_exercise_days, exercise_days, no_of_assets, w, sim_stock_mat,
                                          batch_size, no_of_epochs, no_of_hidden_nodes, no_of_output_nodes, t, nnet_model)

    price_list.append(price)
    
df = pd.DataFrame({'Runs:' + str(no_of_paths):price_list})
# df.to_csv('df_pv_comonotonic_nn_nodes_' + str(no_of_hidden_nodes) + '_sim' + str(no_of_paths) + '.csv')
print(df)
# text_file = open('European_Arithmetic.txt', "a")
# strx = "\n Paths: "+ str(no_of_paths)+", Nodes: "+str(no_of_hidden_nodes)+", Price: " + str(price*Notional)
# text_file.write(strx)
# text_file.close()
