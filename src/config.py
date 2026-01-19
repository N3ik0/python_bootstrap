from pathlib import Path

# --- Configuration du Projet ---

# Chemins de base (Pathlib permet une gestion propre des chemins sous tous les OS)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "saved_models"

# Création du dossier pour les modèles s'il n'existe pas
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Chemins des fichiers de données
TRAIN_IMAGES_PATH = DATA_DIR / "train-images.idx3-ubyte"
TRAIN_LABELS_PATH = DATA_DIR / "train-labels.idx1-ubyte"
TEST_IMAGES_PATH = DATA_DIR / "t10k-images.idx3-ubyte"
TEST_LABELS_PATH = DATA_DIR / "t10k-labels.idx1-ubyte"

# Chemins de sauvegarde des modèles
CNN_MODEL_PATH = MODELS_DIR / "mnist_cnn_model.keras"
KNN_MODEL_PATH = MODELS_DIR / "mnist_knn_model.joblib"
