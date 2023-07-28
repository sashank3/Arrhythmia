
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Bidirectional, LSTM, GRU, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE 
import random


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#



df_train = pd.read_csv("C:/Users/Sashank Reddy/Desktop/College/7th SEM Project/WORKING_MAIN_1/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("C:/Users/Sashank Reddy/Desktop/College/7th SEM Project/WORKING_MAIN_1/mitbih_test.csv", header=None)



Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

X = np.reshape(X,[-1, X.shape[1]*X.shape[2]])
    
ls = {1: 5000, 3: 4000 }

sm = BorderlineSMOTE(sampling_strategy= ls, k_neighbors=6, random_state=12)
X_new, Y_new = sm.fit_sample(X, Y)

temp = list(zip(X_new, Y_new))
random.shuffle(temp)
X_new, Y_new = zip(*temp)
Y_new = np.asarray(Y_new)

X_new = np.reshape(X_new, [len(X_new), 187, 1])


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#



def get_model():
    nclass = 5
    inp = Input(shape=(187, 1))
    # img_1 = Convolution1D(16, kernel_size=3, activation=activations.relu, padding="same")(inp)
    # img_1 = Convolution1D(16, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    # img_1 = MaxPool1D(pool_size=2)(img_1)
    # img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="same")(inp)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(64, kernel_size=5, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution1D(64, kernel_size=5, activation=activations.relu, padding="same")(img_1)    
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(64, kernel_size=7, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution1D(64, kernel_size=7, activation=activations.relu, padding="same")(img_1)    
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    
    # img_1 = MaxPool1D(pool_size=2)(img_1)
    # img_1 = Dropout(rate=0.1)(img_1)    
    # # img_1 = Bidirectional(GRU(32, return_sequences=True, dropout = 0.2))(img_1)
    # img_1 = Bidirectional(GRU(64, return_sequences=True, dropout = 0.2))(img_1)
    # img_1 = GlobalMaxPool1D()(img_1)
    # img_1 = Dropout(rate=0.2)(img_1)
    
    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


model = get_model()

history = model.fit(X_new, Y_new, epochs=5, verbose=2, validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])

pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y_test, pred_test, average=None) 
print("Test f1 score : %s "% f1)

f1 = f1_score(Y_test, pred_test, average="macro")
print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)

conf_mat = confusion_matrix(Y_test, pred_test)

print(conf_mat)

#model.save("C:/Users/Sashank Reddy/Desktop/College/7th SEM Project/WORKING_MAIN_1/saved_model/my_model_cnn_smote_SAME&kern")







