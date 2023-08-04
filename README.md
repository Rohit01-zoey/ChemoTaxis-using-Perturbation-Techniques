# ChemoTaxis-using-Perturbation-Techniques

We employ weight perturbation to train a generic recurrent network. The task at hand is to simulate the C. Elegans. Currently, hand crafted networks are required to simulate it. Weight perturbation helps us to train unstructured networks like RNNs which cannot be trained by techniques like backpropagation.

## Weight perturbation

Weight perturbation is a relatively simple technique to compute parameter gradients. It only needs a forward pass of the network to be possible. Thus, it does not make any assumption on the structure of the network and thus, is much more a versatile method compared to present day methods. Cons obviously include slower compute times but weight perturbation sees a huge oppurtunity for parallelization where each gradient in theory, could be computed on different threads in parallel, thus bringing down the overall training times.

## Structure of the Repository

The structure of the repository follows:

```ruby
project/
├── main.py
├── logger.py
├── network.py
├── config.py
├── config.yml
├── test.py
├── metrics
│   ├── __init__.py
│   ├── utils.py
├── data/
│   ├── sine_wave.py
│   ├── ...
│   └── ...
├── models/
│   ├── __init__.py
│   ├── rnn_1hl.py
│   ├── ...
│   └── ...
├── metrics/
│   ├── __init__.py
│   ├── utils.py
│   ├── ...
├── modules/
│   ├── __init__.py
│   ├── early_stopping.py
│   ├── fsm.py
│   ├── ...
├── optimizer/
│   ├── __init__.py
│   ├── adam.py
│   ├── rmsprop.py
│   ├── ...
├── experiment1/
│   ├── __init__.py
│   ├── experiment1.md
│   ├── ...
│   └── ...
└── README.md
```

## Explanation of the code structure

* logger.py : includes the code for the logger. Logs the training and validation accuracies and loss for every epoch along with the time.
* network.py : inlcudes the code for training of models, weight perturbation, bptt(to be implemented) and other helper functions to enable the aforementioned.
* config files :
  * config.py : includes the code for the reading the config file which contains all the hyperparameters for the training step.
  * config.yml : yml file containing for the hyperparameters(set by the user or via command prompt(funtionality to be added))
* metrics > utils.py : contains all the helper functions like softmax, accuracy, mse_loss etc
* data > sine_wave.py : loads the sine wave data
* models > rnn_1hl.py : implements the 1 hiddel layer RNN.
* modules :
  * early_stopping.py : to be implemented
  * fsm.py : sets the implemented RNN in an FSM mode. Thus, the RNN is able to generate the entire sequence based of one sample
* optimizer : to be implemented/contains bugs!
* experiment1 : to be implemented
* main.py : runs the RNN training on the sine data.

> Simply, running the main.py (for now) should yield the results.
> Code will be updated to add for reproducibiliity in the code and experiments

---

CHECK DATALOADER and PLOTTER functions
