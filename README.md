# NLP_GroupFinalProject

Use `requirements.txt` to get all of the necessary libraries to run the model. You can use `pip install -r requirements.txt` to do so.

There is also the option of using an anaconda virtual environment. For this, you have to use `conda` commands with the file `requirements_conda.yml` as follows:

```bash
conda env create -f requirements_conda.yml
```

and to activate the new environment:

```bash
conda activate NLP_tfa
```

**_NOTE:_** In case you would like to perform the model training with GPU support on your machine, make sure that you do that on a Linux machine/emulation. The `tensorflow` package stopped GPU support for Windows machines after the release of version `2.11` and installing it by `python -m pip install tensorflow<2.11` did not succeed at the moment of development of this project (apparently `pip` allows installing newer versions).

## Key components

Everything we have currently can be defined in the following files:  
* `main.py` -- the part where the main control loop of the project takes place: from receiving the data set, to training the model and consequently testing it.
* `architecture.py` -- here we define the architectures of our Encoder and Decoder methods.
* `training.py` -- in this part, the our LSTM with attention architecture, including its functionality for training and code generation is defined.
* `dataloader.py` -- the code within this file is used to receive, pre-process (e.g., clean, tokenize and create emebeddings) our data set.
* `evaluation_metrics.py` -- this file is used for defining all the evalution metrics that will be used to assess the quality of the generated code.
