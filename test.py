#%%
import keras
from keras.models import Sequential
import os
import calls

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float").reshape((-1,28*28))/255.
x_test = x_test.astype('float').reshape((-1,28*28))/255.
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
#%%
model = keras.models.Sequential([
    keras.layers.Dense(51,activation='relu',input_dim=28*28),
    keras.layers.Dense(10,activation='softmax')
])
ckpt_dir = "./ckpt/weights.{epoch}-{batch}.h5"
model.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])
model.summary()

savecall = calls.SaveCall(ckpt_dir,period=300,mode=calls.SaveCall.train_mode)
iepoch = savecall.load(model)

model.fit(x_train,y_train,batch_size=128,epochs=60,verbose=2,callbacks=[savecall],initial_epoch=iepoch)

print(model.evaluate(x_train,y_train,batch_size=128))