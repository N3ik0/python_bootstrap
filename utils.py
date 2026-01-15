import numpy as np
import matplotlib.pyplot as plt

""" Affiche les statistiques (images par chiffre & pourcentage) """
def display_statistics(labels):
    print('--- Distribution des chiffres ---')
    for i in range(10):
        count = np.sum(labels == i)
        percentage = (count / len(labels)) * 100
        print(f"Chiffre {i} : {count} images ({percentage:.2f}%)")

""" Affiche un tableau de répartition des données de test & d'entrainement """
def display_statistics_array(train_label, test_label):
    print('Chiffre | Train Count (%) | Test Count (%)')
    print('------------------------------------------')
    for i in range(10):
        # Compte le bon dataset
        count_test = np.sum(test_label == i)
        count_train = np.sum(train_label == i)
        # Calcul le pourcentage
        train_percentage = (count_train / len(train_label)) * 100
        test_percentage = (count_test / len(test_label)) * 100
        # Affichage
        print(f' {i}      | {count_train} ({train_percentage:.2f}%) | {count_test} ({test_percentage:.2f}%)')

""" Affiche une image du nombre souhaité """
def display_image(loader, number):
    print(f'---Affichage de l\'image : {number} ---')
    # Créé un "mask" pour définir quel nombre on veut
    mask = (loader.y_train == number)
    # Choisi l'image à partir du mask décidé (image[masque])
    image_to_display = loader.x_train[mask]
    # Moyenne de toute les images
    mean_image = np.mean(image_to_display, axis=0)
    # Affichage de la moyenne des images (cmap en param du type de couleurs souhaité)
    plt.imshow(mean_image, cmap="magma")
    # Ajout d'un titre à l'affichage
    plt.title(f"Moyenne du chiffre : {number}")
    # Affiche l'image
    plt.show()

""" Affiche tous les nombres en une fois """
def display_all_numbers(loader):
    # Défini la taille de la sortie du graph
    plt.figure(figsize=(15,5))
    for i in range(10):
        # Défini l'affichage (2, 5, i+1), un tableau de 2 lignes, 5 colonnes et ou commencer (i+1)
        plt.subplot(2, 5, i+1)
        # Création d'un masque pour définir quel chiffre utiliser
        mask = (loader.y_train == i)
        # Calcul la moyenne du chiffre en cours
        mean_img = np.mean(loader.x_train[mask], axis = 0)
        # Paramètre l'affichage (data, couleurs)
        plt.imshow(mean_img, cmap="magma")
        # Ajoute un titre à la visualisation
        plt.title(f"Chiffre {i}")
    # Recalcul pour eviter les chevauchements de texte
    plt.tight_layout()
    plt.show()


""" Affiche la répartition en graphique barres """
def display_distribution_bar_chart(train_labels):
    count = []
    for i in range(10):
        train = np.sum(train_labels == i)
        count.append(train)
    bars = plt.bar(range(10), count)
    plt.xticks(range(10))
    plt.gca().bar_label(bars, padding=3)
    plt.margins(y=0.1)
    plt.title("Données d'entrainement en barres")
    plt.xlabel("Chiffres")
    plt.ylabel("Occurences")


