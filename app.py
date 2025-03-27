import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow import keras
from flask import Flask, request, render_template, jsonify


app = Flask(__name__)


md = keras.models.load_model("model_autoencoder.keras")




sclr = MinMaxScaler()


def pre_data(df):
    for cl in df:
        df[cl] = df[cl].astype(str)
    df.drop(columns=["Timestamp"], inplace=True)
    df_scl = sclr.fit_transform(df)
    return df_scl


def loadfile():
    fl = request.files["file"]
    if not fl:
        return jsonify({"Error!": "File not Uploaded!"}), 400
    
    df = pd.read_csv(fl)
    pro_data = pre_data(df)

    reconst = md.predict(pro_data)
    merr = np.mean(np.abs(pro_data - reconst))
    thresh = np.percentile(merr, 90)
    anomlay = (merr > thresh).astype(int)

    df["Anomaly"] = anomlay
    res = df.to_dict(orient="records")

    return jsonify({"Logs": res})



@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)