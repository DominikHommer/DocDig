from .module_base import Module
from typing import List, Dict
import numpy as np

class CellFormatter(Module):
    def __init__(self):
        super().__init__("cell-formatter")

    def get_preconditions(self) -> List[str]:
        return ['cell-denoiser']

    def process(self, data: dict, config: dict) -> List[Dict]:
        pages = data["cell-denoiser"]
        output = []

        for page in pages:
            formatted_page = {"columns": []}

            for column in page["columns"]:
                formatted_column = []

                for cell_image in column:

                    # Interpolate, and normalize image
                    image = (cell_image - cell_image.min()) / (cell_image.max() - cell_image.min())
                    image = 1.0 - image
                    image = (image * 255).astype(np.uint8)

                    formatted_column.append({
                        "image": image,
                        "erkannt": "",
                        "score": -1,
                        "verbesserung": "",
                        "skip_ocr": False
                    })

                formatted_page["columns"].append(formatted_column)

            output.append(formatted_page)

        return output
