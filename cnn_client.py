import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
import numpy as np

class CNNClient:
    def __init__(self, path) -> None:
        self.path = path
    
    def load_and_setup_cnn(self, img_height, img_width, batch_size, dataset_dir):
        self.model = load_model(self.path)
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2)
        self.train_generator = self.train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')
        return self.model
    
    def preprocess_img(self, path, show_img = False):
        img = image.load_img(path, target_size=(22, 150))
        if show_img:
            display(img)
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        img_array /= 255.0
        return img_array
    
    def predict(self, img_to_predict):
        predictions = self.model.predict(img_to_predict)
        index_to_class = {v: k for k, v in self.train_generator.class_indices.items()}
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = index_to_class[predicted_class_index]
        return predicted_class_name
    
    
    ## TODO: modular für Text und Zahlenerkennung