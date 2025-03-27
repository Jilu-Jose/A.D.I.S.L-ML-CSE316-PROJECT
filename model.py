import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv("log_data.csv")
df.head(10)
df.tail(10)

df.describe()
df.info()
df.dropna()


lab_enc = {}
for cl in ["Threat Type", "Severity", "Status"]:
    le = LabelEncoder()
    df[cl] = le.fit_transform(df[cl])
    lab_enc[cl] =le

df.drop(columns=["Timestamp"], inplace=True)
X = df.drop(columns=["Anomaly"])
y = df["Anomaly"]

sclr = MinMaxScaler()
X_sclr = sclr.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_sclr, y, test_size=0.2, random_state = 42)

in_dim = X_train.shape[1]


autoenc = keras.Sequential([
    layers.Input(shape=(in_dim,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(in_dim, activation="sigmoid")
])


autoenc.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

autoenc.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test), verbose=1)

X_test_predic = autoenc.predict(X_test)
merr = np.mean(np.abs(X_test - X_test_predic), axis = 1)

thresh = np.percentile(merr, 90)

y_predic = (merr > thresh).astype(int)


autoenc.save("model_autoencoder.keras")





