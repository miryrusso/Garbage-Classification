# Garbage-Classification

## Overview

Lo scopo di questo progetto è quello di allenare un modello affinchè sappia classificare i rifiuti in 6 diverse classi(**glass, paper, cardboard, plastic, metal, trash**). 

## Dataset

È stato utilizzato un dataset kaggle [**Garbage Classification**](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data) (≈2.5 k immagini, ritagliate 384x384). 
Esso viene automaticamente scaricato dataset all'avvio del file *main.ipynb*.

## Layout Progetto

```
project-garbage-classifier/
│
├── README.md          ← *you are here*
├── requirements.txt   ← pip dependencies
│
├── database/
│   └── database.py    ← dataset download / CSV utilities
│
├── src/
│   ├── pre_processing.py  ← data transforms
│   ├── classifier.py      ← model, training e evaluation
│   └── main.py            ← command‑line entry point
│
└── .gitignore
```

## Quick Start

```bash
# 1. create & activate virtual env
python3 -m venv .venv && source .venv/bin/activate

# 2. install deps
pip install -r requirements.txt

# 3. demo
cd demo
python3 demo.py
```

## Relazione

La rlazione si può trovare in *docs/relazione.pdf*
