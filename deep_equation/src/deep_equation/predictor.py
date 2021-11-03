"""
Predictor interfaces for the Deep Learning challenge.
"""

from typing import List
import numpy as np
from PIL import Image
import cv2
from deep_equation.model import MyModel
import torch
from torchvision import transforms
import os.path

class BaseNet:
    """
    Base class that must be used as base interface to implement 
    the predictor using the model trained by the student.
    """

    def load_model(self, model_path):
        """
        Implement a method to load models given a model path.
        """
        pass

    def predict(
        self, 
        images_a: List, 
        images_b: List, 
        operators: List[str], 
        device: str = 'cpu'
    ) -> List[float]:
        """
        Make a batch prediction considering a mathematical operator 
        using digits from image_a and image_b.
        Instances from iamges_a, images_b, and operators are aligned:
            - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
        Args: 
            * images_a (List[PIL.Image]): List of RGB PIL Image of any size
            * images_b (List[PIL.Image]): List of RGB PIL Image of any size
            * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
                - invalid options must return `None`
            * device: 'cpu' or 'cuda'
        Return: 
            * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
                [{digit from image_a} {operator} {digit from image_b}]
        """
    # do your magic

    pass 


class RandomModel(BaseNet):
    """This is a dummy random classifier, it is not using the inputs
        it is just an example of the expected inputs and outputs
    """

    def load_model(self, model_path):
        """
        Method responsible for loading the model.
        If you need to download the model, 
        you can download and load it inside this method.
        """
        np.random.seed(42)

    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ) -> List[float]:

        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):            
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        
        return predictions


class StudentModel(BaseNet):
    """
    TODO: THIS is the class you have to implement:
        load_model: method that loads your best model.
        predict: method that makes batch predictions.
    """

    _to_size = 64
    modelo = None
    validOp = ['+', '-', '*', '/']

    # TODO
    def load_model(self, model_path: str):
        """
        Load the student's trained model.
        TODO: update the default `model_path` 
              to be the correct path for your best model!
        """

        self.classes = [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.11, 0.12, 0.14, 0.17, 0.2, 0.22, 0.25, 0.29, 0.33, 0.38, 0.4, 0.43, 0.44, 0.5, 0.56, 0.57, 0.6, 0.62, 0.67, 0.71, 0.75, 0.78, 0.8, 0.83, 0.86, 0.88, 0.89, 1.0, 1.12, 1.14, 1.17, 1.2, 1.25, 1.29, 1.33, 1.4, 1.5, 1.6, 1.67, 1.75, 1.8, 2.0, 2.25, 2.33, 2.5, 2.67, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 21.0, 24.0, 25.0, 27.0, 28.0, 30.0, 32.0, 35.0, 36.0, 40.0, 42.0, 45.0, 48.0, 49.0, 54.0, 56.0, 63.0, 64.0, 72.0, 81.0, 'NAN']

        self.num_classes = len(self.classes)
        self.class_dict = dict(zip(range(self.num_classes), self.classes))

        self.transf = transforms.ToTensor()
        self.modelo = MyModel(self.num_classes)
        self.modelo.load_state_dict(torch.load(model_path))
        self.modelo.eval()

        return self.modelo
        
    
    def _resize_image(self, pil_image):
        size = self._to_size
        pil_image = pil_image.resize((size, size), Image.ANTIALIAS)

        return pil_image
    
    def _chgBack(self, pil_image):
        
        w, h = pil_image.size
        np_img = np.array(pil_image)        
        
        # Detecta se background branco
        if cv2.countNonZero(np_img) > ((w*h)//2):
            np_img = cv2.bitwise_not(np_img)
    
        return Image.fromarray(np_img)
    
    def _binarize_array(self, pil_image, threshold=85):
        """Binarize a numpy array."""
        
        numpy_array = np.array(pil_image)
        
        for i in range(len(numpy_array)):
            for j in range(len(numpy_array[0])):
                if numpy_array[i][j] > threshold:
                    numpy_array[i][j] = 255
                else:
                    numpy_array[i][j] = 0
        
        return Image.fromarray(numpy_array)
    
    def _preprocess(self, image):
        '''Recebe uma PIL image e:
            1 - Converte para escala de cinza
            2 - Aplica fundo preto
            3 - Resize 64 x 64
            4 - Binariza
            5 - Transforma em tensor
        '''
        #Converte para PIL Image
        #image = self.transfPIL(np_image)
        
        #Para escala de cinza
        image = image.convert('L')
        
        #Ajusta background
        image = self._chgBack(image)
        
        #Padroniza o tamanho 64 x 64
        image = self._resize_image(image)
        
        #Binariza a imagem
        image = self._binarize_array(image)
        
        #Transforma num tensor
        tensor_image = self.transf(image)
        
        return tensor_image

    def _posProcess(self, input):
        '''Transforma de classe resultante do modelo para valor''' 
        
        return self.class_dict[input]

    def _opToTensor(self, operator):

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
        
        return result

    # TODO:
    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ):
        """Implement this method to perform predictions 
        given a list of images_a, images_b and operators.
        """
        predictions = []

        if not self.modelo:
            basedir = os.path.abspath(os.path.dirname(__file__))
            localsalvo = os.path.join(basedir, '.', 'best_models', 'MTMModel.pth')
            self.modelo = self.load_model(localsalvo)

        for image_a, image_b, operator in zip(images_a, images_b, operators):    

            image_a = self._preprocess(image_a).unsqueeze(0)  
            image_b = self._preprocess(image_b).unsqueeze(0)  
            operator = self._opToTensor(operator)

            output = self.modelo(image_a, image_b, operator.unsqueeze(0))
            output = output.detach().numpy().argmax()
            output = self._posProcess(output)

            predictions.append(output)
        
        return predictions
