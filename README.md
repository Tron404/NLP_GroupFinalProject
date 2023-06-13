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
---
**_NOTE:_**  

In case you would like to perform the model training with GPU support on your machine, make sure that you do that on a Linux machine/emulation. The `tensorflow` package stopped GPU support for Windows machines after the release of version `2.11` and installing it by `python -m pip install tensorflow<2.11` did not succeed at the moment of development of this project (apparently `pip` allows installing newer versions).
---