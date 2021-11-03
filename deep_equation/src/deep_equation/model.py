import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import math
import cv2
from time import time
import os.path


#Macro variáveis
num_batches = 64
to_size = 64
n_epochs = 30
log_interval = 10

#Classe de coleta dos dados
class NumbsToMath(Dataset):

    def __init__(self, split, tam=None):
        # TODO
        # Get image list and labels
        
        self.classes = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81, 'NAN']
        self.validOp = ['+', '-', '*', '/']
        self.transf = transforms.ToTensor() ## 1 x 512

        self.num_classes = len(self.classes)
        self.class_dict = dict(zip(self.classes, range(self.num_classes)))
        
        if split == 'treino':
            data = datasets.MNIST('./MNIST_data/', download=True, train=True) # Carrega a parte de treino do dataset
        elif split == 'teste':
            data = datasets.MNIST('./MNIST_data/', download=True, train=False) # Carrega a parte de validação do dataset
        else:
            return None

        if tam == None:
            self.tam = len(data)
        else:
            self.tam = tam

        self.files = data

    def _getClass(self, number):
        last = -9

        if number == 'NAN':
            return 'NAN'
        
        for elemento in self.classes:
            if elemento == 'NAN':
                return 'NAN'            
            elif elemento >= float(number):
                return last
            else:
                last = elemento

    def _mathCalculation(self, numA, numB, op):

        if op == '+':
            return self._getClass(numA + numB) 
        elif op == '-':
            return self._getClass(numA - numB) 
        elif op == '*':
            return self._getClass(numA * numB) 
        else:
            if numB != 0:
                return self._getClass("{:.2f}".format(numA / numB))
            else:
                return 'NAN'

    def _toTensor(self, operator):

        if operator not in self.validOp:
            return None

        if operator == '+':
            result = [0.1, 0, 0, 0]
        elif operator == '-':
            result = [0, 0.1, 0, 0]
        elif operator == '*':
            result = [0, 0, 0.1, 0]
        elif operator == '/':
            result = [0, 0, 0, 0.1]

        result = torch.tensor(result)
        #print(result.shape)
        return result
    
    def _resize_image(self, pil_image):
        size = to_size
        pil_image = pil_image.resize((size, size), Image.ANTIALIAS)

        return pil_image
    
    def chgBack(self, pil_image):
        
        w, h = pil_image.size
        np_img = np.array(pil_image)        
        
        # Detecta background branco
        if cv2.countNonZero(np_img) > ((w*h)//2):
            np_img = cv2.bitwise_not(np_img)
    
        return Image.fromarray(np_img)
    
    def binarize_array(self, pil_image, threshold=85):
        """Binarize a numpy array."""
        
        numpy_array = np.array(pil_image)
        
        for i in range(len(numpy_array)):
            for j in range(len(numpy_array[0])):
                if numpy_array[i][j] > threshold:
                    numpy_array[i][j] = 255
                else:
                    numpy_array[i][j] = 0
        
        return Image.fromarray(numpy_array)
    
    def preprocess(self, image):
        '''Recebe uma NP image e:
            1 - Converte para escala de cinza
            2 - Aplica fundo preto
            3 - Resize
            4 - Binariza
            5 - Transforma em tensor
        '''
        #Converte para PIL Image
        #image = self.transfPIL(np_image)
        #image = Image.fromarray(np_image)
        
        #Para escala de cinza
        #image = image.convert('L')
        
        #Ajusta background
        #image = self.chgBack(image)
        
        #Padroniza o tamanho 64 x 64
        image = self._resize_image(image)
        
        #Binariza a imagem
        #image = self.binarize_array(image)
        
        #Transforma num tensor
        tensor_image = self.transf(image)
        
        return tensor_image

    def __getitem__(self, idx):
        # TODO
        # Load images individually and process it with image transformations
        
        #print(idx)
        a = math.floor(idx/(self.tam * len(self.validOp)))
        part_op = math.floor(idx/self.tam)
        op = part_op - a*len(self.validOp)
        b = idx - part_op*self.tam

        #print('a = ', a)
        imageA = self.preprocess(self.files[a][0])
        imageB = self.preprocess(self.files[b][0])
        operator = self.validOp[op]

        label_numA = self.files[a][1]
        label_numB = self.files[b][1]

        #label = resultado da operação entre os dois números
        label = self._mathCalculation(label_numA, label_numB, operator)
        label_int = self.class_dict[label]
        operator_num = self._toTensor(operator)

        return imageA, imageB, operator_num, label_int

    def __len__(self):
        return int(10*self.tam*len(self.validOp))

#Classe da rede convolucional para classificação de dígitos escritos a mão
#Classe treinada em separado para transfer learning
class ConvModel(nn.Module):
    '''
    Modelo de classificação de imagens do MNIST
    '''

    def __init__(self, num_classes=1):
        super().__init__()
        # cria a arquitetura, e inicializa os pesos

        #Part 1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #1x64x64 -> 10x60x60
        self.relu1 = nn.ReLU(inplace=True)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2) #-> 10x30x30
        
        #Part 2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #10x30x30 -> 20x26x26
        self.relu2 = nn.ReLU(inplace=True)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 20x13x13 Stride=4 temporariamente, tem que ser 2
        
        #Part 3
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5) #20x13x13 -> 20x9x9
        self.relu3 = nn.ReLU(inplace=True)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 20x4x4
        
        self.dropout = nn.Dropout2d()
        
        feat_part1Conv = (to_size - 5)/1 + 1
        feat_part1Pool = int((feat_part1Conv - 2)/2 + 1)
        
        feat_part2Conv = int(feat_part1Pool - 5)/1 +1
        feat_part2Pool = int((feat_part2Conv - 2)/2) + 1
        
        feat_part3Conv = int((feat_part2Pool - 5)/1) +1
        feat_part3Pool = int((feat_part3Conv - 2)/2 +1)
        
        num_features = int(feat_part3Pool * feat_part3Pool)*20
        #print(num_features)
        
        self.fc1 = nn.Linear(num_features, 50) 
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, imageA):
        # faz uma predicao
        # importante ter certeza que os shapes dos tensores 
        # são compatíveis com as camadas

        x = self.conv1(imageA) 
        x = self.relu1(x)
        x = self.maxPool1(x)
        
        x = self.conv2(x) 
        x = self.relu2(x)
        x = self.maxPool2(x)
        
        x = self.conv3(x) 
        x = self.relu3(x)
        x = self.maxPool3(x)
        
        x = self.dropout(x)
        #print(x.shape)
        x = x.view(x.shape[0], -1) #Flatten
        
        x = self.fc1(x)
        x = self.fc2(x)

        return x

#Classe que carrega módulo classificador de dígitos e agrega mais 04 camadas lineares
#para operação matemática
class MyModel(nn.Module):

    def __init__(self, num_classes=55):
        super().__init__()
        # cria a arquitetura, e inicializa os pesos
        self.transfPIL = transforms.ToPILImage()
        
        basedir = os.path.abspath(os.path.dirname(__file__))
        localsalvo = os.path.join(basedir, 'best_models', 'ConvModel.pth')

        self.convNet1 = ConvModel()
        self.convNet1.load_state_dict(torch.load(localsalvo))
        self.convNet1.eval()
        self.convNet2 = ConvModel()
        self.convNet2.load_state_dict(torch.load(localsalvo))
        self.convNet2.eval()

        #Entrada do operador: lista de 0 e 1 para cada op: +, -, *, /
        self.fc1 = nn.Linear(2*10 + 4, 128)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(128, num_classes)

        
    def forward(self, imageA, imageB, operator):
        '''
        - Recebe duas PIL Images e um operador como entradas
        - Retorna uma valor de predição de regressão
        '''
        
        # faz uma predicao
        # importante ter certeza que os shapes dos tensores 
        # são compatíveis com as camadas

        xA = self.convNet1(imageA)
        xB = self.convNet2(imageB)
        x = torch.cat((xA, operator, xB), dim=1)
        # print('X: ', x.shape)

        x = self.fc1(x)
        x = self.relu1(x) #Relu
        x = self.fc2(x)
        x = self.relu2(x) #Relu
        x = self.fc3(x)
        x = self.relu3(x)
        pred = self.fc4(x)

        return pred

#Função de treino do modelo de classificação
def treino(modelo, trainloader, optimizer, loss, train_losses, train_counter, epoch):
    
    otimizador = optimizer # define a polítca de atualização dos pesos e da bias    
    criterio = loss # definindo o criterio para calcular a perda
    modelo.train() # ativando o modo de treinamento do modelo
    perda_acumulada = 0 # inicialização da perda acumulada da epoch em questão

    for batch_idx, (imageA, imageB, operator, label) in enumerate(trainloader):
        otimizador.zero_grad() # zerando os gradientes por conta do ciclo anterior
        output = modelo(imageA, imageB, operator) # colocando os dados no modelo
        #print('Tempo predição: ', time() - inicio)

        perda_instantanea = criterio(output, label) # calculando a perda da epoch em questão
        perda_instantanea.backward() # back propagation a partir da perda
        #print('Tempo backward: ', time() - inicio)

        otimizador.step() # atualizando os pesos e a bias
        perda_acumulada += perda_instantanea.item() # atualização da perda acumulada    
        #print('Tempo otimizador: ', time() - inicio) 
        
        if batch_idx % log_interval == 0:
        
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(imageA), len(trainloader.dataset),
            100. * batch_idx / len(trainloader), perda_instantanea.item()))
    
            train_losses.append(perda_instantanea.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(trainloader.dataset)))
        
    print("Epoch {} - Perda resultante: {} - Loss batch: {}".format(epoch+1, perda_acumulada/len(trainloader), perda_instantanea))

#Função de teste/avaliação do modelo 
def test(modelo, test_loader, test_losses, test_counter):
    modelo.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for imageA, imageB, operator, target in test_loader:
            output = modelo(imageA, imageB, operator)
            loss = nn.CrossEntropyLoss()
            test_loss += loss(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return (100. * correct / len(test_loader.dataset))


def trainRun():
  #Sequência de comandos principais para treino do modelo

  #Construção do set de treino
  train_set = NumbsToMath('treino', 1000)
  train_loader = DataLoader(train_set, batch_size=num_batches, shuffle=True)
  dataiter = enumerate(train_loader)
  print(type(next(dataiter)))

  test_losses = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

  #Criação do modelo com número de classes definidas no dataset de treino
  modelo = MyModel(len(train_set.classes))
  print(modelo)

  # Definição de loss e optimizer
  #optimizer = optim.Adam(model.parameters(), lr=1e-3)
  param = list(modelo.fc1.parameters()) + list(modelo.fc2.parameters()) + list(modelo.fc3.parameters()) + list(modelo.fc4.parameters())
  optimizer = optim.Adam(param, lr=1e-3)
  loss = nn.CrossEntropyLoss()

  #Criação do dataset de avaliação
  teste_set = NumbsToMath('teste', 100)
  print(teste_set)
  test_loader = DataLoader(teste_set, batch_size=num_batches, shuffle=True)

  inicio = time() 
  train_losses = []
  train_counter = []
  
  saveAcc = 0
  for epoch in range(1, n_epochs + 1):
      treino(modelo, train_loader, optimizer, loss, train_losses, train_counter, epoch)
      acc = test(modelo, test_loader, test_losses, test_counter)
      if acc > saveAcc:
        torch.save(modelo.state_dict(), './modelMTM2.pth')
        torch.save(optimizer.state_dict(), './optimizerMTM2.pth')

      
  print("\nTempo de treino (em minutos) =",(time()-inicio)/60)

#trainRun()