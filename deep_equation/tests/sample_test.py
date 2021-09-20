import unittest
from PIL import Image
from deep_equation import predictor


class TestRandomModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.digit_a = Image.open('resources/digit_a.png')
        self.digit_b = Image.open('resources/digit_b.png')

        self.input_imgs_a = [
            self.digit_a, self.digit_a, self.digit_b, self.digit_b, self.digit_a]
        self.input_imgs_b = [
            self.digit_b, self.digit_b, self.digit_a, self.digit_b, self.digit_a]
        self.operators = ['+', '-', '*', '/', '*']

    def test_random_predictor(self):
        """
        Test random prediction outputs. 
        """
        basenet = predictor.RandomModel()

        output = basenet.predict(
            self.input_imgs_a, 
            self.input_imgs_b, 
            operators=self.operators, 
            device='cpu',
        )

        self.validate_output(output)
    
    def test_student_predictor(self):
        """
        Test student prediction outputs. 
        """

        basenet = predictor.StudentModel()

        output = basenet.predict(
            self.input_imgs_a, 
            self.input_imgs_b, 
            operators=self.operators, 
            device='cpu',
        )

        self.validate_output(output)

    def validate_output(self, output):
        """
        Validate output format.
        """

        # Make sure we got one prediction per input_sample
        self.assertEqual(len(output), len(self.input_imgs_a))
        self.assertEqual(len(self.input_imgs_b), len(self.input_imgs_a))
        self.assertEqual(type(output), list)

        # Make sure that that predictions are floats and not other things
        self.assertEqual(type(float(output[0])), float)
        
        # Ensure that the output range is approximately correct
        for out in output:
            self.assertGreaterEqual(out, -10)
            self.assertLessEqual(out, 100)
