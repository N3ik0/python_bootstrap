from loader import Loader as loader
import utils
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Création du loader
loader = loader("./data")
# Chargement des données dans le loader
loader.load()

print(f"Nombres d'images chargées {len(loader.x_train)}")

mask = (loader.y_train == 5)
how_much = np.sum(mask)
print(how_much)

# Distribution des différents chiffres
utils.display_statistics(loader.y_train)
# Distribution differentes via array
utils.display_statistics_array(loader.y_train, loader.y_test)

utils.display_distribution_bar_chart(loader.y_train)

# Affichage du chiffre souhaité 
# utils.display_image(loader, 5)

# Affichage de tous les chiffres
utils.display_all_numbers(loader)

# reshape des images
reshape_img = loader.prepare_dataset()
# Optimisation des images pour le calcul (passage à 1)

# Création du modèle
knn = KNeighborsClassifier(n_neighbors=3)
# Entrainement du modèle
print('---Entrainement en cours---')
knn.fit(loader.x_train, loader.y_train)
# Prédiction du modèle
y_pred = knn.predict(loader.x_test)
# Calcul du score entre 0 et 1
score = accuracy_score(loader.y_test, y_pred)
error_indices = np.where(y_pred != loader.y_test)[0]
print(f"Le nombres total d'erreurs est : {len(error_indices)}")
print(f"L'accuracy du modèle est de : {score * 100:.2f}%")

# Affichage de la matrice de confusion
cm = utils.display_confusion_matrix(loader.y_test, y_pred)