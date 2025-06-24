
import numpy as np
import pytesseract

from .module_base import Module


def crop_center(image, width, height):
    h, w = image.shape[:2]
    x_start = max((w - width) // 2, 0)
    y_start = max((h - height) // 2, 0)
    return image[y_start:y_start + height, x_start:x_start + width]

class DetectSpeciesColumn(Module):
    def __init__(self,
                 debug: bool = False,
                 keywords: list[str] = ["espèce", "espece", "art"]):
        super().__init__("spezies-spalten-markierer")
        self.debug = debug
        self.keywords = set(k.lower() for k in keywords)

    def get_preconditions(self) -> list[str]:
        return ["row-extractor"]

    def process(self, data: dict, config: dict) -> list[dict]:
        pages = data["row-extractor"]
        output = []

        for page_idx, page in enumerate(pages):
            processed_page = {"columns": []}

            for col_idx, column_cells in enumerate(page["columns"]):
                is_spezies = False

                if not column_cells or not isinstance(column_cells[0], np.ndarray):
                    processed_page["columns"].append({
                        "cells": [],
                        "is_spezies_spalte": False
                    })
                    continue

                first_cell_img = column_cells[0]
                # 1. Cropped Center Test
                cropped = crop_center(first_cell_img, width=200, height=50)
                text_cropped = pytesseract.image_to_string(cropped, lang="fra").strip().lower()

                if any(k in text_cropped for k in self.keywords):
                    is_spezies = True
                else:
                    # 2. Full Cell Test
                    text_full = pytesseract.image_to_string(first_cell_img, lang="fra").strip().lower()
                    if any(k in text_full for k in self.keywords):
                        is_spezies = True
                    else:
                        # 3. Fallback: Zellbreite
                        if first_cell_img.shape[1] >= 280:
                            is_spezies = True

                # Umwandeln der np.ndarrays zu dicts mit image key
                cell_dicts = [{"image": img, "skip_ocr": False} for img in column_cells]

                processed_page["columns"].append({
                    "cells": cell_dicts,
                    "is_spezies_spalte": is_spezies
                })

            output.append(processed_page)

        return output
