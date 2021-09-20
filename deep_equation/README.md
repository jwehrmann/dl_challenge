# Deep Equation Challenge

Package with the submission interface. It has a RandomModel so you can see the expected inputs and outputs. Also, it has a test to validate the students implementation.

The package is pip-installable, it is very easy to update it and implement the predictor for the student best model trained. 

## Requirements

* python3.7 ou superior

> Outros requisitos que seu modelo precisar deverão ser adicionados no arquivo `requirements.txt` com a versão correta!

## Install

Abra um terminal e, estando no mesmo diretório deste arquivo e do `setup.py`, rode:

```
pip install -e .
```

Pronto, você já tem o pacote instalado. 

## Test

To test all models:
```
pytest tests/
```

To test only the random model (example):
```
pytest tests/ -k test_random -s
```
> `-k` flag allows filtering some tests
> `-s` shows the print outputs, useful for debugging

To test only the model implemented by the student:
```
pytest tests/ -k test_student -s
```


## Note

The `model.py` and `train.py` files are not necessary for the submission, though, the student can use those files (and create other ones, if needed) to run everything inside this package. 
