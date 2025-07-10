# Garbage-Classification

## Autori
Questo progetto è stato realizzato da Giada Margarone e Miriana Russo

## Overview

Lo scopo di questo progetto è quello di allenare un modello affinchè sappia classificare i rifiuti in 6 diverse classi: **glass, paper, cardboard, plastic, metal, trash**. 

## Dataset

È stato utilizzato un dataset kaggle [**Garbage Classification**](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data) (≈2.5 k immagini, ritagliate 384x384). 
Il dataset viene automaticamente scaricato all'avvio del file *main.ipynb*.

## Layout Progetto

```
project-garbage-classifier/
│
├── README.md          
├── requirements.txt   
│
├── database/
│   └── database.py    
│
├── src/
│   ├── pre_processing.py 
│   ├── classifier.py      
│   └── main.py           
│
└── .gitignore
```
## Dependencies 
Assicurati di avere git lfs installato.

## Demo quick start

```bash
# 1. Clona la repository
git clone https://github.com/miryrusso/Garbage-Classification.git
```

```bash
# 2. Entra nella cartella
cd Garbage-Classification && git lfs pull
```

```bash
# 3. crea e attiva il virtual-env
python3 -m venv .venv && source .venv/bin/activate
```
```bash
# 4. installa le dipendenze
pip install -r requirements.txt
```
```bash
# 5. demo
cd demo
python3 demo.py
```

https://github.com/user-attachments/assets/23d15f5a-9557-4c70-8739-056b043f612e





## Relazione

La relazione finale del progetto è disponibile al seguente [**link**](https://github.com/miryrusso/Garbage-Classification/tree/main/docs)
