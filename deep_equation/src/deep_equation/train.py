'''
Teste de leitura dos dígitos
'''

from predictor import StudentModel as SM
from pathlib import Path
from PIL import Image
from glob import glob # busca arquivos no disco 

dataset_path = r'C:\Users\dayso\OneDrive\Documents\Mestrado PUC-Rio\2021.2\Aprendizado profundo\Atv Extra\dl_challenge\deep_equation\resources'
files = glob(f'{dataset_path}/*.png') # pego o path de todas as imagens do teste
toPredictA, toPredictB, operators = [], [], []
validOP = ['+', '-', '*', '/']

for image1 in files:
    for image2 in reversed(files):
        for operator in validOP:
            image_pathA = Path(image1)
            toPredictA.append(Image.open(image_pathA))

            image_pathB = Path(image2)
            toPredictB.append(Image.open(image_pathB))

            operators.append(operator)
    
predictor = SM()
output = predictor.predict(toPredictA, toPredictB, operators)

for numA, numB, op, result in zip(toPredictA, toPredictB, operators, output):
    #numA.show()
    #numB.show()
    print(f'{op} == {result}')