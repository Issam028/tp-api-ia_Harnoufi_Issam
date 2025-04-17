import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Lecture des fichiers ===

def lire_fichier_csv(chemin_fichier, skip_header=True):
    data_csv = []
    try:
        with open(chemin_fichier) as fic:
            lines = fic.readlines()
            data_csv = [line.strip().split(";") for line in lines]
            if skip_header:
                data_csv = data_csv[1:]
    except FileNotFoundError:
        print(f"File {chemin_fichier} not found!")
    return data_csv

data_usagers = lire_fichier_csv("data/usagers-2023.csv")

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

# Suppression des lignes qui sont trop courtes
data_usagers = [d for d in data_usagers if len(d) > 8]

# Conversion des données
xy = [
    [
        convert_annee(d[8][1:-1]),
        convert_grav(d[6][1:-1])
    ]
    for d in data_usagers
]

# Elimination des données incohérentes
xy = [d for d in xy if d[0] > -1 and d[1] > -1]

print(f"Taille des données en entrée : {len(data_usagers)}")
print(f"Taille des données après filtrage : {len(xy)}")

if len(xy) == 0:
    print("No valid data available for training!")
    exit()

x_annee = [xy[0] for xy in xy]
y_gravite = [xy[1] for xy in xy]

# === répartition des données ===
x_train, x_test, y_train, y_test = train_test_split(
    x_annee, y_gravite, test_size=0.3, random_state=42
)

# === normalisation ===
scaler = StandardScaler()

x_train = np.array(x_train).reshape(-1, 1)
x_test = np.array(x_test).reshape(-1, 1)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# == Choix de la stratégie ===
model = LinearRegression()

# === Entrainnement ===
model.fit(x_train, y_train)

# === Vérification ===
y_pred = model.predict(x_test)

# Save the model and scaler for later use
joblib.dump(model, 'model_accident.pkl')
joblib.dump(scaler, 'scaler_accident.pkl')
print("Model and scaler saved successfully")