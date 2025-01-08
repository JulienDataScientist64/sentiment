import os
import pickle
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Définir les chemins des fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../models/lstm.h5"))
TOKENIZER_PATH = os.path.abspath(os.path.join(BASE_DIR, "../models/tokenizer.pkl"))

# Charger le modèle
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Modèle chargé avec succès.")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

# Charger le tokenizer
try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("[INFO] Tokenizer chargé avec succès.")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du tokenizer : {e}")

# Initialisation de l'application FastAPI
app = FastAPI()

# Modèle de données pour la validation
class DataInput(BaseModel):
    data: List[str]

# Endpoint racine
@app.get("/")
def root():
    return {"message": "API is running"}

# Endpoint de prédiction
@app.post("/predict")
def predict(input_data: DataInput):
    try:
        # Prétraitement des textes
        sequences = tokenizer.texts_to_sequences(input_data.data)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)

        # Générer les prédictions
        predictions = model.predict(padded_sequences).tolist()
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")
