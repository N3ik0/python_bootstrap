import tensorflow as tf
import numpy as np
import os
from src.data_loader import DataLoader
from src.models.cnn_model import create_cnn_model
from src.config import CNN_MODEL_PATH
import src.utils as utils

def train_cnn_pipeline():
    """
    Orchestre le processus complet d'entraînement du CNN :
    1. Chargement et préparation des données
    2. Création (ou chargement) du modèle
    3. Entraînement
    4. Sauvegarde
    5. Évaluation
    """
    print("--- 1. Préparation des données ---")
    data_loader = DataLoader()
    # Charge et prépare les données pour le CNN (reshape 4D + normalisation)
    data_loader.preprocess_cnn()

    # Vérifie si un modèle existe déjà (optionnel, ici on repart souvent de zéro ou on continue)
    if CNN_MODEL_PATH.exists():
        print(f"--- Modèle existant trouvé sous {CNN_MODEL_PATH}. Chargement... ---")
        model = tf.keras.models.load_model(CNN_MODEL_PATH)
    else:
        print("--- Création d'un nouveau modèle CNN ---")
        model = create_cnn_model(input_shape=(28, 28, 1))

    # Configuration de l'arrêt anticipé (Early Stopping)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    print("\n--- 2. Démarrage de l'entraînement ---")
    history = model.fit(
        data_loader.x_train, 
        data_loader.y_train, 
        epochs=100,  # Max epochs, stoppé avant par early_stop
        validation_data=(data_loader.x_test, data_loader.y_test),
        callbacks=[early_stop]
    )

    print(f"\n--- 3. Sauvegarde du modèle sous {CNN_MODEL_PATH} ---")
    model.save(CNN_MODEL_PATH)

    print("\n--- 4. Évaluation et Visualisation ---")
    # Performance sur le jeu de test
    results = model.evaluate(data_loader.x_test, data_loader.y_test, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]*100:.2f}%")

    # Matrice de confusion
    print("Génération de la matrice de confusion...")
    predictions_prob = model.predict(data_loader.x_test)
    y_pred = np.argmax(predictions_prob, axis=1)
    
    utils.display_confusion_matrix(data_loader.y_test, y_pred, title="Matrice de Confusion CNN (Fin d'entraînement)")

    return model, history
