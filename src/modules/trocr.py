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

        # Pick from valid previous module outputs
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
                processed_column = []

                if col_idx == 0:

                    for cell in column:

                        if cell["skip_ocr"]:
                            processed_column.append(cell)

                        else: # cell["skip_ocr"] is false
                            image = cell["image"]
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                            pil_image = Image.fromarray(image).convert("RGB")
                            #print(f"Image dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
                            #cv2.imshow("Quotation Mark Candidate", image)
                            #cv2.waitKey(0)  # Press a key to close
                            #cv2.destroyAllWindows()

                            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

                            with torch.no_grad():
                                generated_ids = self.model.generate(**inputs)
                                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                            cell["erkannt"] = text
                            cell["score"] = 0

                            processed_column.append(cell)
                            print(f"Text erkannt: {text}")

                else: # For other columns
                    processed_column.extend(column)

                processed_page["columns"].append(processed_column)

            output.append(processed_page)

        return output
