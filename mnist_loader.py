# Loader MNIST
from pathlib import Path
import numpy as np

"""
X_train = Images d'entrainements
Y_train = Labels d'entrainements
X_test = Images de test
y_test = Labels de tests
"""

class MNISTLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        if self.data_dir.exists():
            print(f"Le dossier trouvé à {self.data_dir.resolve()} existe")
        else:
            print("Le dossier trouvé n'existe pas")

    def load(self):
       # Chargement des images d'entrainement
        with open(self.data_dir / "train-images.idx3-ubyte", "rb") as f:
            data = np.fromfile(f, dtype=np.uint8, offset=16)
            self.X_train = data.reshape(-1, 28, 28)

        # Chargement des labels d'entraînements 
        with open(self.data_dir / "train-labels.idx1-ubyte", "rb") as f:
            self.y_train = np.fromfile(f, dtype=np.uint8, offset=8)
        
        # Chargement des images de test
        with open(self.data_dir / "t10k-images.idx3-ubyte", "rb") as f:
            self.X_test = np.fromfile(f, dtype=np.uint8, offset=16)

        # Chaargeement des labels de test
        with open(self.data_dir / "t10k-labels.idx1-ubyte", "rb") as f:
            self.y_test = np.fromfile(f, dtype=np.uint8, offset=8)
        
        print("Chargement terminé !")

