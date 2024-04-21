import tensorflow as tf
from tensorflow import keras
from keras.layers import Input,Dense,Normalization
from keras.models import Sequential
import pandas as pd
import datetime

class PerceptronMulticapaK:
    def __init__(self, x):
        self.input = Normalization(input_shape=x.shape[1:])
        self.hidden=Dense(40,activation="linear")
        self.hidden2=Dense(40,activation="linear")
        self.output=Dense(1,activation="linear")
        self.model=Sequential()
        self.model.add(self.input)
        self.model.add(self.hidden)
        self.model.add(self.hidden2)
        self.model.add(self.output)
        self.optimizer=keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(loss='mse', optimizer=self.optimizer,metrics = ["RootMeanSquaredError"])

    def train(self,x,y,xv,yv):
        tag=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/fit/" + tag
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.input.adapt(x)
        self.model.fit(x, y, validation_data=(xv,yv),epochs=128,callbacks=[tensorboard_callback])
        self.model.save('modelos/model.keras')
        yv_pred=self.model.predict(xv)
        pd.DataFrame(yv_pred).to_csv("yv_pred.csv",index=False,sep=";")
        pd.DataFrame(yv).to_csv("yv.csv",index=False,sep=";")