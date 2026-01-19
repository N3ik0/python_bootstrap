from mnist_loader import MNISTLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def print_stats(labels, name="Dataset"):
    print(f"\nDistribution pour {name}: ")
    digits, counts= np.unique(labels, return_counts=True)
    for digit, count in zip(digits, counts):
        print(f"Chiffre {digit}: {count} images")

def display_mean_digit(loader, digit):
    # On récupère toutess les images qui correspondent au chiffre "digit"
    mask = (loader.y_train == digit)
    images_du_chiffre = loader.X_train[mask]

    # On calcule la moyenne sur le premier axe (le nombre d'images)
    # axis=0 permet d'écraser les 6000 images de 28x28 en une seule de 28x28
    mean_image = np.mean(images_du_chiffre, axis=0)
    
    plt.imshow(mean_image, cmap='magma') # 'magma' ou 'gray', au choix ! [cite: 23]
    plt.title(f"Moyenne du chiffre : {digit}")
    plt.show()

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 8))
    # annot = True affiche les nombres, fmt="d" évite l'écriture scientifique
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Vrais Labels')
    plt.title('Matrice de Confusion')
    plt.show()