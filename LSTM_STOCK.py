import tushare as ts
import datetime
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

class Stock:
    def __init__(self,code,balance=10000,price = 1.0,hold = 0):
        self.code = code
        self.balance = balance
        self.price = price
        self.hold = hold
        self.total = balance + price * hold

    def update(self,price=0.0):
        self.price = price
        self.total = self.balance + self.hold*price

    def buy(self):
        price = self.price
        inc_hold = np.floor(self.balance/price)
        self.hold +=  inc_hold
        self.balance -= inc_hold*price

    def sell(self):
        price = self.price
        hold = self.hold
        self.balance += hold*price
        self.hold = 0

    def __str__(self):
        return 'Trading:\ncode = %s\nbalance = %d\nprice = %f\nhold = %d\ntotal = %d'%(self.code,self.balance,self.price,self.hold,self.total)


def get_data(code,end='2017-01-01', duration=365):
    d1 = datetime.datetime.strptime(end,'%Y-%m-%d')
    d2 = d1 - datetime.timedelta(days=duration)
    start = d2.strftime('%Y-%m-%d')
    return ts.get_h_data(code, start=start, end=end)

def train_test_split(data,SEQ_LENGTH = 25,test_prop=0.3):
    data = data.sort_index()
    ntrain = int(len(data) *(1-test_prop))
    predictors = data.columns[:4]
    #norms = data[predictors].apply(np.linalg.norm)
    data_pred = data[predictors] #/norms
    num_attr = data_pred.shape[1]
    result = np.empty((len(data) - SEQ_LENGTH - 1, SEQ_LENGTH, num_attr))
    y = np.empty(len(data) - SEQ_LENGTH - 1)
    yopen = np.empty(len(data) - SEQ_LENGTH - 1)

    for index in range(len(data) - SEQ_LENGTH - 1):
        result[index, :, :] = data_pred[index: index + SEQ_LENGTH]
        y[index] = data.iloc[index + SEQ_LENGTH + 1].close
        yopen[index] = data.iloc[index + SEQ_LENGTH + 1].open

    xtrain = result[:ntrain, :, :]
    ytrain = y[:ntrain]
    xtest = result[ntrain:, :, :]
    ytest = y[ntrain:]
    ytest_open = yopen[ntrain:]
    return xtrain, xtest, ytrain, ytest, ytest_open


def train_model(xtrain,ytrain,SEQ_LENGTH=25,N_HIDDEN=256):
    # SEQ_LENGTH = 25  # Sequence Length, or # of days of trading
    # N_HIDDEN = 256  # Number of units in the hidden (LSTM) layers
    # num_attr = 4  # Number of predictors used for each trading day
    num_attr = xtrain.shape[2]
    model = Sequential()
    model.add(LSTM(N_HIDDEN, return_sequences=True, activation='tanh', input_shape=(SEQ_LENGTH, num_attr)))
    model.add(Dropout(0.2))
    model.add(LSTM(N_HIDDEN, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer='adam')  ## optimizer = 'rmsprop'
    model.fit(xtrain, ytrain, batch_size=50, nb_epoch=50, validation_split=0.05)
    return model

def predict(model,xtest):
    predicted = model.predict(xtest)
    return predicted

def plot(y_test,predicted):
    plt.plot(y_test, label='true values')
    plt.plot(predicted, label='Predictions')
    plt.legend()
    plt.show()


def policy(code,xtest,ytest,ytest_open,model):
    ypred = model.predict(xtest)

    ### the first day
    xnow = xtest[0]
    price = xnow[-1,2]
    stock = Stock(code,price=price)
    pred_price = ypred[0,0]
    totals = [stock.total]

    for i in range(1,len(xtest)):
        price_open = ytest_open[i]
        price_close = ytest[i]
        stock.update(price=price_open)
        pred_price_now = ypred[i,0]
        if pred_price_now >= pred_price:
            stock.buy()
        else:
            stock.sell()
        pred_price = pred_price_now
        stock.update(price=price_close)
        totals.append(stock.total)

    plt.plot(totals)
    plt.title('Wealth curve')
    plt.show()
    return totals


def back_test(self):
    pass


def profit_plot(self):
    pass

if __name__ == '__main__':
    code = '300222' ## 科大智能
    df = ts.get_stock_basics()
    print df.ix[code][['name', 'industry', 'timeToMarket']]

    stock = Stock(code)
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    data = get_data(code, end=today, duration=1000)

    plt.plot(data.close)
    plt.show()
    xtrain, xtest, ytrain, ytest, ytest_open = train_test_split(data)

    model = train_model(xtrain,ytrain)


    predicted_tr = model.predict(xtrain)
    plt.plot(ytrain, label='true values')
    plt.plot(predicted_tr, label='predicted  values')
    plt.legend()
    #plt.show()

    predicted_test = model.predict(xtest)
    plt.plot(ytest, label='true values')
    plt.plot(predicted_test, label='predicted  values')
    plt.legend()
    #plt.show()

    totals = policy(code, xtest, ytest, ytest_open, model)





