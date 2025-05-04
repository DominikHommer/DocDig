import os
import numpy as np
import cv2
from .module_base import Module
from tensorflow.keras.models import load_model
from typing import List

class Predictor(Module):
    def __init__(self, model_path: str = "./config/classifier_model.keras", target_size=(80, 384), class_labels: List[str] = None):
        super().__init__("predictor")
        self.model_path = model_path
        self.model = load_model(model_path)
        self.target_size = target_size  # Muss zum Trainingsinput des Modells passen
        self.class_labels = class_labels  # Optional: z. B. ['A', 'B', ..., 'Z']

    def get_preconditions(self) -> List[str]:
        return ['cell-denoiser']

    def process(self, data: dict, config: dict) -> List[str]:
        cleaned_images = data['cell-denoiser']
        predictions = []

        for img in cleaned_images:
            if img is None:
                predictions.append(None)
                continue

            img_resized = cv2.resize(img, self.target_size)
            img_resized = np.expand_dims(img_resized, axis=-1)  # Kanal hinzufügen
            img_resized = np.expand_dims(img_resized, axis=0)   # Batch-Dimension
            img_resized = img_resized.astype("float32") / 255.0

            pred = self.model.predict(img_resized)
            pred_class = np.argmax(pred, axis=1)[0]

            if self.class_labels:
                predictions.append(self.class_labels[pred_class])
            else:
                predictions.append(pred_class)

        return predictions
