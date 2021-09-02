# Deep Equation Challenge

Pacote com a interface padrão para a entrega. 

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
