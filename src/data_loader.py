import numpy as np
from src.config import TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, TEST_IMAGES_PATH, TEST_LABELS_PATH

class DataLoader:
    """
    Classe responsable du chargement des datasets MNIST.
    Elle isole la complexité liée à la lecture des fichiers binaires.
    """
    def __init__(self):
        # Images(X) et Labels(Y) pour l'entraînement et le test
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load(self):
        """ Charge les fichiers binaires bruts en mémoire """
        
        # Vérification sommaire (optionnelle, mais bonne pratique)
        if not TRAIN_IMAGES_PATH.exists():
            raise FileNotFoundError(f"Fichier introuvable: {TRAIN_IMAGES_PATH}")

        # Chargement des images d'entrainement
        with open(TRAIN_IMAGES_PATH, "rb") as f:
            # offset=16 car l'en-tête du fichier idx3 fait 16 octets
            data = np.fromfile(f, dtype=np.uint8, offset=16)
            self.x_train = data.reshape(-1, 28, 28)
            print(f"Chargé x_train: {self.x_train.shape}")

        # Chargement des labels d'entrainement
        with open(TRAIN_LABELS_PATH, "rb") as f:
            # offset=8 car l'en-tête du fichier idx1 fait 8 octets
            self.y_train = np.fromfile(f, dtype=np.uint8, offset=8)
            print(f"Chargé y_train: {self.y_train.shape}")

        # Chargement des images de test
        with open(TEST_IMAGES_PATH, "rb") as f:
            data = np.fromfile(f, dtype=np.uint8, offset=16)
            self.x_test = data.reshape(-1, 28, 28)
            print(f"Chargé x_test: {self.x_test.shape}")

        # Chargement des labels de test
        with open(TEST_LABELS_PATH, "rb") as f:
            self.y_test = np.fromfile(f, dtype=np.uint8, offset=8)
            print(f"Chargé y_test: {self.y_test.shape}")

    def preprocess_flatten(self):
        """ 
        Prépare le dataset pour des modèles simples (ex: KNN, Dense Neural Net).
        Aplatit les images (28x28 -> 784) et normalise (0-255 -> 0-1).
        """
        if self.x_train is None: self.load()
        self.x_train = self.x_train.reshape(-1, 784) / 255.0
        self.x_test = self.x_test.reshape(-1, 784) / 255.0

    def preprocess_cnn(self):
        """ 
        Prépare le dataset pour un CNN.
        Garde la structure 2D et ajoute le canal de couleur : (28, 28, 1).
        Normalise les valeurs (0-255 -> 0-1).
        """
        if self.x_train is None: self.load()
        self.x_train = self.x_train.reshape(-1, 28, 28, 1) / 255.0
        self.x_test = self.x_test.reshape(-1, 28, 28, 1) / 255.0
