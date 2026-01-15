from loader import Loader as loader
import utils
import numpy as np


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

# Affichage du chiffre souhaité 
# utils.display_image(loader, 5)

# Affichage de tous les chiffres
utils.display_all_numbers(loader)