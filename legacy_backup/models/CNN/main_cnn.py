from pathlib import Path
import sys
import os
import numpy as np

# 1. Gestion du chemin pour importer le loader et utils
# On remonte à la racine du projet pour trouver loader.py et utils.py
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from loader import Loader as loader
import utils
import tensorflow as tf
from tensorflow.keras import layers, models

# 2. Préparation des données
data_loader = loader('./data')
data_loader.load()
data_loader.prepare_cnn_dataset()

# Nom du fichier où est stocké le "cerveau" entraîné
model_path = 'mnist_cnn_model.keras'

# 3. Logique de chargement ou d'entraînement
if os.path.exists(model_path):
    print(f"\n--- Modèle trouvé ({model_path}). Chargement en cours... ---")
    # On charge le modèle existant sans rien calculer
    model = tf.keras.models.load_model(model_path)
else:
    print("\n--- Aucun modèle trouvé. Début de la construction et de l'entraînement... ---")
    
    # Construction de l'architecture
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        
        # Extraction (Convolution + Pooling)
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Classification (Flatten + Dense)
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Configuration de l'apprentissage
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Ajout d'early stopping pour s'arrêter quand le val loss arrête de progresser
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Entraînement (X passages sur les données, avec early_stop)
    model.fit(
        data_loader.x_train, 
        data_loader.y_train, 
        epochs=100, 
        validation_data=(data_loader.x_test, data_loader.y_test),
        callbacks=[early_stop]
    )

    # Sauvegarde pour la prochaine fois
    model.save(model_path)
    print(f"Modèle sauvegardé sous {model_path}")

# 4. Analyse des performances (Matrice de Confusion)
print("\n--- Génération de la matrice de confusion ---")

# On demande au modèle de prédire sur les données de test
# predict() renvoie les probabilités (ex: 0.98 pour le chiffre 5)
predictions_prob = model.predict(data_loader.x_test)

# On transforme les probabilités en l'indice du chiffre le plus probable (np.argmax)
y_pred = np.argmax(predictions_prob, axis=1)

# On utilise ton utilitaire pour afficher le graphique
# y_true = data_loader.y_test (la vérité)
# y_pred = ce que l'IA a trouvé
utils.display_confusion_matrix(data_loader.y_test, y_pred)

# 5. Résumé final
model.summary()

# 6. Evaluation du modèle chargé
print(f"-----Évaluation du modèle chargé-----")
results = model.evaluate(data_loader.x_test, data_loader.y_test, verbose=0)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]*100:.2f}%")