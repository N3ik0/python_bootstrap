import tensorflow as tf
import numpy as np
from src.data_loader import DataLoader
from src.config import CNN_MODEL_PATH
import src.utils as utils

def predict_pipeline():
    """
    Lance le pipeline de prédiction :
    1. Charge les données (Test)
    2. Charge le modèle entraîné
    3. Effectue des prédictions
    4. Affiche les résultats
    """
    if not CNN_MODEL_PATH.exists():
        print(f"Erreur : Le modèle {CNN_MODEL_PATH} n'existe pas. Veuillez lancer l'entraînement d'abord.")
        return

    print("--- 1. Chargement des données de test ---")
    data_loader = DataLoader()
    # On a besoin du preprocess CNN pour avoir la bonne forme (28, 28, 1)
    data_loader.preprocess_cnn()

    print(f"--- 2. Chargement du modèle depuis {CNN_MODEL_PATH} ---")
    model = tf.keras.models.load_model(CNN_MODEL_PATH)

    print("--- 3. Évaluation rapide ---")
    results = model.evaluate(data_loader.x_test, data_loader.y_test, verbose=0)
    print(f"Précision sur le jeu de test : {results[1]*100:.2f}%")

    print("--- 4. Visualisation des prédictions ---")
    # On prédit sur tout le test set
    predictions_prob = model.predict(data_loader.x_test)
    y_pred = np.argmax(predictions_prob, axis=1)

    # Affichage Matrice Confusion
    utils.display_confusion_matrix(data_loader.y_test, y_pred, title="Matrice de Confusion (Mode Prédiction)")
    
    # Exemple : Afficher quelques erreurs
    print("Recherche d'erreurs pour visualisation...")
    errors = np.where(y_pred != data_loader.y_test)[0]
    if len(errors) > 0:
        # On affiche la première erreur trouvée
        idx_err = errors[0]
        true_val = data_loader.y_test[idx_err]
        pred_val = y_pred[idx_err]
        print(f"Exemple d'erreur : Image index {idx_err} (Vrai: {true_val}, Prédit: {pred_val})")
        # Note: Pour afficher via utils.display_image, il faudrait adapter car l'utilitaire attend 
        # la structure du loader brut, mais ici x_test a été reshape.
        # On peut adapter manuellement si nécessaire ou créer une fonction specifique.
    else:
        print("Aucune erreur trouvée sur le test set ! (Bravo)")

if __name__ == "__main__":
    predict_pipeline()
