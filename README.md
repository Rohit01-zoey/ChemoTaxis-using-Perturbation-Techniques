# ChemoTaxis-using-Perturbation-Techniques
We employ weight perturbation to train a generic recurrent network. The task at hand is to simulate the C. Elegans. Currently, hand crated networks are required to simulate it.

## Structure of the Repository

The structure of the repository follows:
```
project/
├── main.py
├── logger.py
├── network.py
├── metrics
│   ├── __init__.py
│   ├── utils.py
├── data/
│   ├── dataset.py
│   ├── preprocessing.py
│   └── ...
├── models/
│   ├── __init__.py
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── experiments/
│   ├── experiment1/
│   │   ├── config.json
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── visualize.py
│   │   └── ...
│   ├── experiment2/
│   │   ├── config.json
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── visualize.py
│   │   └── ...
│   └── ...
└── README.md
```