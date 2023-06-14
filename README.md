# ChemoTaxis-using-Perturbation-Techniques
We employ weight perturbation to train a generic recurrent network. The task at hand is to simulate the C. Elegans. Currently, hand crated networks are required to simulate it.

The structure of the repository follows:
```
project/
├── main.py
├── logger.py
├── network.py
├── utils.py
├── data/
│   ├── dataset.py
│   ├── preprocessing.py
│   └── ...
├── models/
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