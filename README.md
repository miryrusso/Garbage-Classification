# Garbage-Classification

## Autori
Questo progetto è stato realizzato da Giada Margarone e Miriana Russo

## Overview

Lo scopo di questo progetto è quello di allenare un modello affinchè sappia classificare i rifiuti in 6 diverse classi(**glass, paper, cardboard, plastic, metal, trash**). 

## Dataset

È stato utilizzato un dataset kaggle [**Garbage Classification**](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data) (≈2.5 k immagini, ritagliate 384x384). 
Esso viene automaticamente scaricato dataset all'avvio del file *main.ipynb*.

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

## Quick Start

```bash
# 1. crea e attiva i;lvirtual env
python3 -m venv .venv && source .venv/bin/activate

# 2. installa le dipendenze
pip install -r requirements.txt

# 3. demo
cd demo
python3 demo.py
```

## Relazione

La rlazione si può trovare in *docs/relazione.pdf*
