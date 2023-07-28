# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:51:26 2020

@author: Sashank Reddy
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


ip = np.load(r"C:\Users\Sashank Reddy\Desktop\College\7th SEM Project\WORKING_MAIN_1\data\Data8\271.npy")
plt.plot(ip)
ip_new = []

for i in range(0,187):
    ip_new.append([ip[i]])
    
ip = np.array([ip_new])

reconstructed_model = keras.models.load_model("C:/Users/Sashank Reddy/Desktop/College/7th SEM Project/WORKING_MAIN_1/saved_model/my_model_cnn_smote_SAME&kern")
# reconstructed_model.summary()
res = reconstructed_model.predict(ip)
res = np.argmax(res, axis=-1)
print(res)