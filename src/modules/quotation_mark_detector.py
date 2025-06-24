from .module_base import Module
from typing import List, Dict
import cv2
import numpy as np

class QuotationMarkDetector(Module):
    def __init__(self, threshold=0.05):
        super().__init__("quotationmark-detector")
        self.threshold = threshold

    def get_preconditions(self) -> List[str]:
        return ['cell-formatter']

    def detect_quotation_marks(self, image, relative_non_white_threshold: float) -> bool:
        # Interpolation and normalization already done in cell_formatter.py
        #image = (image - image.min()) / (image.max() - image.min())
        #image = 1.0 - image
        #image = (image * 255).astype(np.uint8)

        non_white_pixels = np.sum(image < 250)  # allow small tolerance
        total_pixels = image.shape[0] * image.shape[1]
        relative_non_white = non_white_pixels / total_pixels

        return relative_non_white < relative_non_white_threshold

    def process(self, data: dict, config: dict) -> List[Dict]:
        pages = data["cell-formatter"]
        if isinstance(pages, dict):
            pages = [pages]

        output = []

        for page in pages:
            processed_page = {"columns": []}

            for column in page["columns"]:
                cells = column["cells"]
                is_spezies_spalte = column.get("is_spezies_spalte", False)
                processed_cells = []

                if not is_spezies_spalte:
                    processed_page["columns"].append(column)
                    continue

                for cell in cells:
                    image = cell["image"]
                    is_quote = self.detect_quotation_marks(image, self.threshold)

                    if is_quote:
                        print("Detected QuotationMark")

                    cell["erkannt"] = '"' if is_quote else cell.get("erkannt", "")
                    cell["score"] = 100 if is_quote else cell.get("score", -1)
                    cell["skip_ocr"] = is_quote

                    processed_cells.append(cell)

                processed_page["columns"].append({
                    "cells": processed_cells,
                    "is_spezies_spalte": True
                })

            output.append(processed_page)

        return output
