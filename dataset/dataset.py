import os
import zipfile
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
from torch.utils import data
from os.path import join
from PIL import Image


# Percorsi relativi, dato che sei già nella cartella "dataset"
zip_path = "./dataset/archive.zip"
extract_dir = "./dataset/garbage_dataset"

# Controlla se il file zip esiste
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Dataset estratto con successo.")
else:
    print("Dataset già presente.")

# Dizionario per mappare le classi ai numeri
# Le classi sono le cartelle all'interno della cartella "Garbage classification"
class_dict = {
    'glass': 1,
    'paper': 2,
    'cardboard': 3,
    'plastic': 4,
    'metal': 5,
    'trash': 6
}

# Funzione per ottenere la classe da un percorso
# La funzione prende un percorso e restituisce la classe corrispondente
# La classe è il nome della cartella che contiene l'immagine
def class_from_paths(path):
    class_name = os.path.basename(os.path.dirname(path))
    return class_dict[class_name]

paths = glob('./dataset/garbage_dataset/Garbage classification/Garbage classification/*/*.jpg')
labels = [class_from_paths(p) for p in paths]

# Crea un DataFrame con i percorsi e le etichette
# Il DataFrame avrà due colonne: 'path' e 'label'
# 'path' conterrà i percorsi delle immagini
# 'label' conterrà le etichette corrispondenti
dataset = pd.DataFrame({'path':paths, 'label':labels})

# Funzione per dividere il dataset in train, validation e test
# perc è una lista con le percentuali per train, validation e test
# ad esempio perc=[0.65, 0.15, 0.2]
# dove 65% per il train, il 15% per la validation e il 20% per il test
def split_train_val_test(dataset, perc=[0.65, 0.15, 0.2]):
    train, testval = train_test_split(dataset, test_size = perc[1]+perc[2])
    val, test = train_test_split(testval, test_size = perc[2]/(perc[1]+perc[2]))
    return train, val, test

# Imposta il seed per la riproducibilità
random.seed(123456789)
np.random.seed(123456789)
train, val, test = split_train_val_test(dataset)

# Crea la cartella CSV se non esiste
if not os.path.exists('CSV'):
    os.makedirs('CSV')

# Salva i DataFrame in file CSV
# I file CSV saranno utili per il training del modello
def dataset_to_csv(train, val, test):
    train.to_csv('./dataset/CSV/train.csv', index=None)
    val.to_csv('./dataset/CSV/valid.csv', index=None)
    test.to_csv('./dataset/CSV/test.csv', index=None)

# Classe per il dataset che legge i dati da un file CSV
# La classe eredita da data.Dataset di PyTorch
# La classe prende come input il percorso del dataset, il file CSV e le trasformazioni da applicare alle immagini
class CSVImageDataset(data.Dataset):
    def __init__ (self, data_root, csv, transform = None):
        self.data_root = data_root
        self.data = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_path, image_label = self.data.iloc[i]['path'], self.data.iloc[i].label
        image_label = image_label - 1

        images = Image.open(join(self.data_root,image_path)).convert('RGB')
        if self.transform is not None:
            images = self.transform(images)

        return images, image_label