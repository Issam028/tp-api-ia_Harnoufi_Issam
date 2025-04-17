from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

def lire_fichier_csv(chemin_fichier, skip_header=True):
    data_csv = []
    with open(chemin_fichier) as fic:
        lines = fic.readlines()
        data_csv = [line.strip().split(";") for line in lines]
        if skip_header:
            data_csv = data_csv[1:]
        return data_csv

def convert_grav(val):
    if val == "1":
        return 1
    elif val == "2":
        return 100
    elif val == "3":
        return 10
    elif val == "4":
        return 5
    return -1

def convert_annee(val):
    if len(val) != 4:
        return -1
    return int(val, 10)

def process_data():
    data_usagers = lire_fichier_csv("data/usagers-2023.csv")
    data_usagers = [d for d in data_usagers if len(d) > 8]
    xy = [[convert_annee(d[8][1:-1]), convert_grav(d[6][1:-1])] for d in data_usagers]
    xy = [d for d in xy if d[0] > -1 and d[1] > -1]
    x_annee = [xy[0] for xy in xy]
    y_gravite = [xy[1] for xy in xy]
    return x_annee, y_gravite

def train_model(x_annee, y_gravite):
    x_train, x_test, y_train, y_test = train_test_split(x_annee, y_gravite, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    x_train = np.array(x_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Save the trained model to a file
    joblib.dump(model, 'model_accident.pkl')
    return model, x_test, y_test

def predict_accident(x_test, model):
    y_pred = model.predict(x_test)
    return y_pred


