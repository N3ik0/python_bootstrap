from pathlib import Path
import numpy as np

class Loader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        # Images(X) and Labels(Y)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        if self.data_dir.exists():
            print(f'Data from {self.data_dir} loaded correctly')
        else:
            print(f'The data from {self.data_dir} doesn\'t exist')

    def load(self):
        # Chargement des images d'entrainement
        with open(self.data_dir / "train-images.idx3-ubyte", "rb") as f:
            data = np.fromfile(f, dtype=np.uint8, offset=16)
            self.x_train = data.reshape(-1, 28, 28)
            print(f"x_train shape: {self.x_train.shape}")


        with open(self.data_dir / "train-labels.idx1-ubyte", "rb") as f:
            self.y_train = np.fromfile(f, dtype=np.uint8, offset=8)
            print(f"y_train shape: {self.y_train.shape}") 

        with open(self.data_dir / "t10k-images.idx3-ubyte", "rb") as f:
            data = np.fromfile(f, dtype=np.uint8, offset=16)
            self.x_test = data.reshape(-1, 28, 28)
            print(f"x_test shape: {self.x_test.shape}")

        with open(self.data_dir / "t10k-labels.idx1-ubyte", "rb") as f:
            self.y_test = np.fromfile(f, dtype=np.uint8, offset=8)
            print(f"y_test shape: {self.y_test.shape}")