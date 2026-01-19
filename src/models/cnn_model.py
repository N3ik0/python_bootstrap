import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Construit et compile l'architecture du modèle CNN.
    Cette fonction encapsule la logique de création du réseau de neurones.
    
    Args:
        input_shape: La forme des données d'entrée (hauteur, largeur, canaux).
        num_classes: Le nombre de classes en sortie (10 pour les chiffres 0-9).
        
    Returns:
        model: Un modèle Keras compilé.
    """
    model = models.Sequential([
        # Définition de l'entrée
        layers.Input(shape=input_shape),
        
        # --- Partie Extraction de Caractéristiques (Convolution) ---
        # 32 filtres de taille 3x3
        layers.Conv2D(32, (3, 3), activation='relu'),
        # Réduction de dimension (prend le max sur 2x2 pixels)
        layers.MaxPooling2D((2, 2)),
        
        # 64 filtres de taille 3x3
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # --- Partie Classification ---
        # Aplatit les volumes 3D en vecteurs 1D
        layers.Flatten(),
        # Couche dense pour apprendre les combinaisons non-linéaires
        layers.Dense(64, activation='relu'),
        # Couche de sortie : 1 probabilité par classe
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compilation du modèle avec l'optimiseur et la fonction de perte
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
