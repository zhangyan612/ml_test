from __future__ import print_function
import numpy as np
import pandas as pd
from deep_learning.stocks import backtest as twp
import random, timeit
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.externals import joblib
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop


# save model and run using pre trained model

# determine amount of money to invest
# multiple instruments



# Load data
def load_data():
    np.random.seed(1335)  # for reproducibility
    np.set_printoptions(precision=5, suppress=True, linewidth=150)

    price = np.sin(np.arange(300) / 30.0)  # sine prices
    return price

def load_stock_data():
    prices = pd.read_pickle('data/EURUSD_1day.pkl')
    print('total data set: %s' % len(prices.index))
    x_train = prices.iloc[-650:-200, ]
    print('training data set: %s' % len(x_train.index))
    print('start with %s' % x_train.index[0])
    return x_train


# Initialize first state, all items are placed deterministically
# state is current bar position of xdata, xdata is the collection of state: [current_price, gain/loss], which is the price history
def init_state(data):
    close = data
    diff = np.diff(data)
    diff = np.insert(diff, 0, 0)

    # --- Preprocess data
    xdata = np.column_stack((close, diff))
    xdata = np.nan_to_num(xdata)
    scaler = preprocessing.StandardScaler()
    xdata = scaler.fit_transform(xdata)

    state = xdata[0:1, :]
    return state, xdata

# def init_state(indata):
#     close = indata['Value'].values
#     diff = np.diff(close)
#     diff = np.insert(diff, 0, 0)
#
#     # --- Preprocess data
#     xdata = np.column_stack((close, diff))
#     xdata = np.nan_to_num(xdata)
#     scaler = preprocessing.StandardScaler()
#     # xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
#     # joblib.dump(scaler, 'data/scaler.pkl')
#     # state = xdata[0:1, 0:1, :]
#     xdata = scaler.fit_transform(xdata)
#     state = xdata[0:1, :]
#     return state, xdata

# Take Action
def take_action(state, xdata, action, signal, time_step):
    # this should generate a list of trade signals that at evaluation time are fed to the backtester
    # the backtester should get a list of trade signals and a list of price data for the assett
    # let computer decide how much to buy
    # make necessary adjustments to state and then return it
    time_step += 1

    # if the current iteration is the last state ("terminal state") then set terminal_state to 1
    if time_step == xdata.shape[0]:
        state = xdata[time_step - 1:time_step, :]
        terminal_state = 1
        signal.loc[time_step] = 0
        return state, time_step, signal, terminal_state

    # move the market data window one step forward
    state = xdata[time_step - 1:time_step, :]
    # take action
    if action != 0:
        # signal.loc[time_step] = action
        if action == 1: # buy
            signal.loc[time_step] = 100000
        elif action == 2: # hold
            signal.loc[time_step] = -100000
        elif action == 3: # clear position
            signal.loc[time_step] = 0
    terminal_state = 0

    return state, time_step, signal, terminal_state


# Get Reward, the reward is returned at the end of an episode
def get_reward(new_state, time_step, action, xdata, signal, terminal_state, epoch=0):
    reward = 0
    cash = 0
    holding = 0
    signal.fillna(value=0, inplace=True)
    if terminal_state == 0:
        # get reward for the most current action
        if signal[time_step] != signal[time_step - 1] and terminal_state == 0:
            i = 1
            while signal[time_step - i] == signal[time_step - 1 - i] and time_step - 1 - i > 0:
                i += 1
            reward = (xdata[time_step - 1, 0] - xdata[time_step - i - 1, 0]) * signal[
                time_step - 1] * -100 + i * np.abs(signal[time_step - 1]) / 10.0
        if signal[time_step] == 0 and signal[time_step - 1] == 0:
            reward -= 10

    # calculate the reward for all actions if the last iteration in set
    if terminal_state == 1:
        # run backtest, send list of trade signals and asset data to backtest function
        bt = twp.Backtest(pd.Series(data=[x[0] for x in xdata]), signal, signalType='capital', initialCash = 100000)
        reward = bt.pnl.iloc[-1]
        # cash =  bt.data['cash'][time_step]
        # holding = bt.data['value'][time_step]

    return reward


def evaluate_Q(eval_data, eval_model, epoch_num):
    # This function is used to evaluate the perofrmance of the system each epoch, without the influence of epsilon and random actions
    signal = pd.Series(index=np.arange(len(eval_data)))
    state, xdata = init_state(eval_data)
    status = 1
    terminal_state = 0
    time_step = 1
    while (status == 1):
        # We start in state S
        # Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state.reshape(1, 2), batch_size=1)
        action = (np.argmax(qval))
        # Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        # Observe reward
        eval_reward = get_reward(new_state, time_step, action, xdata, signal, terminal_state, epoch_num)
        state = new_state
        if terminal_state == 1:  # terminal state
            status = 0
    return eval_reward



# def decide_action(cash, holding):
#     # given cash level, decide avaialbel action
#     if cash > 0:
#         action = np.random.randint(1, 2)
#     else:
#         action = np.random.randint(2, 3)

def get_random_action():
    # how many actions -- buy hold sell
    return np.random.randint(0, 4)
    # return random.choice([1,3])

# This neural network is the the Q-function, run it like this:
# model.predict(state.reshape(1,64), batch_size=1)

def init_model():
    model = Sequential()
    model.add(Dense(4, init='lecun_uniform', input_shape=(2,)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2)) I'm not using dropout in this example
    model.add(Dense(4, init='lecun_uniform'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(4, init='lecun_uniform'))
    model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    return model


def run():
    model = init_model()

    start_time = timeit.default_timer() #timer
    indata = load_data() #load_stock_data()
    epochs = 100
    gamma = 0.9  # a high gamma makes a long term reward more valuable
    epsilon = 1
    learning_progress = [] # stores tuples of (S, A, R, S')
    h = 0
    signal = pd.Series(index=np.arange(len(indata)))
    initial_balance = 100000
    available_cash = [100000]
    holding = [0]

    for i in range(epochs):

        state, xdata = init_state(indata)
        status = 1
        terminal_state = 0
        time_step = 1
        # new_holding, new_cash
        # while learning is still in progress
        while (status == 1):
            # We start in state S
            # Run the Q function on S to get predicted reward values on all the possible actions
            qval = model.predict(state.reshape(1, 2), batch_size=1)
            if (random.random() < epsilon) and i != epochs - 1:  # maybe choose random action if not the last epoch
                action = get_random_action()  # assumes 4 different actions
                # action is the random of negtive holding to available cash

                # print('action from random: %s' % action)
            else:  # choose best action from Q(s,a) values
                action = (np.argmax(qval))
                # print('action from qval: %s' % action)

            # Take action, observe new state S'
            new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
            # print(new_state, time_step, signal, terminal_state)
            # Observe reward
            reward = get_reward(new_state, time_step, action, xdata, signal, terminal_state, i)


            # print('reward:%s' % reward)
            # Get max_Q(S',a)
            newQ = model.predict(new_state.reshape(1, 2), batch_size=1)
            maxQ = np.max(newQ)
            y = np.zeros((1, 4))
            y[:] = qval[:]

            if terminal_state == 0:  # non-terminal state
                update = (reward + (gamma * maxQ))
            else:  # terminal state (means that it is the last state)
                update = reward

            y[0][action] = update  # target output
            model.fit(state.reshape(1, 2), y, batch_size=1, nb_epoch=1, verbose=0)
            state = new_state
            if terminal_state == 1:  # terminal state
                status = 0

        eval_reward = evaluate_Q(indata, model, i)
        print("Epoch #: %s Reward: %f Epsilon: %f" % (i, eval_reward, epsilon))
        learning_progress.append((eval_reward))
        # available_cash.append(new_cash)
        # holding.append(new_holding)

        if epsilon > 0.1:
            epsilon -= (1.0 / epochs)

    elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
    print("Completed in %f" % (elapsed,))
    # save the model
    # model.save('ex2_balance.h5')

    # plot results
    bt = twp.Backtest(pd.Series(data=[x[0] for x in xdata]), signal, signalType='shares')
    bt.data['delta'] = bt.data['shares'].diff().fillna(0)
    print(bt.data)
    plt.figure()
    bt.plotTrades()
    plt.suptitle('total epochs:' + str(epochs))
    plt.close('all')
    plt.figure()
    plt.subplot(3, 1, 1)
    bt.plotTrades()
    plt.subplot(3, 1, 2)
    bt.pnl.plot(style='x-')
    plt.subplot(3, 1, 3)
    plt.plot(learning_progress)
    plt.savefig('plt/final_trades_balance' + '.png', bbox_inches='tight', pad_inches=1, dpi=72)  # assumes there is a ./plt dir
    plt.show()



if __name__ == "__main__":
    run()
    # run_with_model()

