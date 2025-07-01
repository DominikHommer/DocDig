from .module_base import Module
import numpy as np
from typing import List, Dict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
import pandas as pd
import cv2

class TrOCR(Module):
    def __init__(self, model_name="microsoft/trocr-base-stage1", output_path="data/output/trocr_output.xlsx"):
        super().__init__("trocr")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.output_path = output_path

    def get_preconditions(self) -> List[str]:
        return ['quotationmark-detector', 'cell-formatter']

    def process(self, data: dict, config: dict) -> List[Dict]:
        valid_keys = self.get_preconditions()
        input_key = next((k for k in valid_keys if k in data), None)
        if input_key is None:
            raise ValueError("No valid input found for TrOCR.")

        pages = data[input_key]
        if isinstance(pages, dict):
            pages = [pages]

        output = []

        for page_idx, page in enumerate(pages):
            processed_page = {"columns": []}

            for col_idx, column in enumerate(page["columns"]):
                is_species_column = column.get("is_species_column", False)
                is_age_column = column.get("is_age_column", False)
                cells = column["cells"]
                processed_column = []

                # Speichere Beispielbild für Debug-Zwecke
                if cells and isinstance(cells[0], dict) and "image" in cells[0] and cells[0]["image"] is not None:
                    cv2.imwrite(f"test_png_{col_idx}.png", cells[0]["image"])

                if not (is_species_column or is_age_column):
                    print(f"Skipping OCR for column {col_idx} – not marked as 'species' or 'age' column.")
                    processed_page["columns"].append(column)
                    continue

                for cell_idx, cell in enumerate(cells):

                    if cell_idx == 0:  # if header
                        processed_column.append(cell)
                        continue

                    if cell["skip_ocr"]:
                        processed_column.append(cell)
                        print("OCR skipped due to quotationmark")
                    else:
                        image = cell["image"]
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        pil_image = Image.fromarray(image).convert("RGB")

                        #print(f"Image dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
                        #cv2.imshow("Quotation Mark Candidate", image)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()

                        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

                        with torch.no_grad():
                            generated_ids = self.model.generate(**inputs)
                            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                        cell["erkannt"] = text
                        cell["score"] = 0
                        processed_column.append(cell)
                        print(f"Text erkannt: {text}")

                processed_page["columns"].append({
                    "cells": processed_column,
                    "is_batch_column": column.get("is_batch_column", False),
                    "is_species_column": is_species_column,
                    "is_age_column": is_age_column
                })

            output.append(processed_page)

        return output

