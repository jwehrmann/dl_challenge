from typing import List
import numpy as np


class BaseNet:

    def load_model(self, model_path):
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
    
    def load_model(self, model_path):
        np.random.seed(42)
    
    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ) -> List[float]:

        predictions = []
        for image_a, image_b in zip(images_a, images_b):
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        
        return predictions


class StudentModel(BaseNet):

    # TODO
    def load_model(self, model_path):
        """
        Load the student's trained model.
        """
        pass
    
    # TODO:
    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ):
        """Implement this method to perform predictions 
        given a list of images_a, images_b and operators.
        """
        predictions = []
        
        return predictions
