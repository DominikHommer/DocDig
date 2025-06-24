import os
import numpy as np
import cv2
from .module_base import Module
import tensorflow
from tensorflow.keras.models import load_model

def weighted_mse(y_true, y_pred):
    img_height, img_width = y_true.shape[1], y_true.shape[2]
    
    x = np.linspace(0, 1, img_width)
    y = np.linspace(0, 1, img_height)
    xv, yv = np.meshgrid(x, y)
    
    sigma = 0.3
    mask = np.exp(-((xv - 0.5)**2 + (yv - 0.5)**2) / (2 * sigma**2))
    mask = tensorflow.convert_to_tensor(mask, dtype=tensorflow.float32)
    mask = tensorflow.expand_dims(mask, axis=-1)
    
    error = tensorflow.square(y_true - y_pred)
    weighted_error = error * mask
    return tensorflow.reduce_mean(weighted_error)

class CellDenoiserResult:
    columns: list[list[np.ndarray]]

class CellDenoiser(Module):
    def __init__(self, debug=False, debug_folder="debug/debug_cell_denoiser/"):
        super().__init__("cell-denoiser")
        
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['row-extractor']
    
    def process(self, data: dict, config: dict) -> list:
        pages: list = data.get('row-extractor')

        model = load_model(config["denoise"]["model"], custom_objects={"weighted_mse": weighted_mse})
    
        result = []
        for p_i, page in enumerate(pages):
            page_data: CellDenoiserResult = {
                "columns": [list() for _ in range(len(page['columns']))]
            }

            for col_nr, col in enumerate(page["columns"]):
                if len(col) == 0:
                    page_data["columns"][col_nr] = []
                    
                    continue

                for row_nr, img in enumerate(col):
                    o_h, o_w = img.shape
                    img_resized = cv2.resize(img, (384, 80))
                    img_resized = np.expand_dims(img_resized, axis=-1)
                    img_resized = np.expand_dims(img_resized, axis=0)

                    output = model.predict(cv2.bitwise_not(img_resized))
                    output = np.squeeze(output)
                    output = cv2.resize(output, (o_w, o_h))

                    if self.debug:
                        cv2.imwrite(f"{os.path.dirname(self.debug_folder)}/page_{p_i}_column_{col_nr}_row_{row_nr}.jpg", output)

                    page_data["columns"][col_nr].append(output) # img for the original image, output for the denoised image

            result.append(page_data)
        
        return result