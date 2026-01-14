#test
from mnist_loader import MNISTLoader
import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# création de l'instance
loader = MNISTLoader("./data")

# Chargement des datas
loader.load()


# utils.display_mean_digit(loader, 9)

# Applatissement des images via reshape
# -1 calcule automatiquement le nb d'image via numpy
X_train = loader.X_train.reshape(-1, 784)
X_test = loader.X_test.reshape(-1, 784)

# reshape a 1 des datas
X_train = X_train / 255.0
X_test = X_test / 255.0



# pas de modifications sur les labels
y_train = loader.y_train
y_test = loader.y_test



# Vérification des dimensions
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# création du model
knn = KNeighborsClassifier(n_neighbors=3)
# entrainement
print("Entrainement en cours")
knn.fit(X_train, y_train)
# prédiction
print("Prédiction en cours")
y_pred = knn.predict(X_test)
# Evaluation : quelle est la précision du modèle
accuracy = accuracy_score(y_test, y_pred)

error_indices = np.where(y_pred != y_test)[0]
print(f"Le nombres total d'erreurs est : {len(error_indices)}")

print(f"Précision du modèle :  {accuracy * 100:.2f}%")

print(f"\n Analyse des erreurs par chiffre")
for num in numbers:
    # isoler les images qui sont réellement le chiffre (y_test == num)
    real_indices = (y_test == num)
    total_numbers = np.sum(real_indices)

    # Parmis celles ci, compter le nombre que l'ia a râté (y_pred != num)
    error_per_number = np.sum((y_test == num) & (y_pred != num))

    # calculer le ratio (erreurs/total)
    error_rate = ((error_per_number / total_numbers) * 100)

    print(f"Chiffre {num} : {error_per_number}/{total_numbers} erreurs {error_rate:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
visual_conf_matrix = utils.plot_confusion_matrix(conf_matrix)
print('\nMatrice de confusion: ')
print(visual_conf_matrix)



# for number in numbers:
#     utils.display_mean_digit(loader, number)