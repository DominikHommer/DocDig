from .module_base import Module
import numpy as np
from typing import List, Dict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
import pandas as pd


class TrOCR(Module):
    def __init__(self, model_name="microsoft/trocr-base-stage1", output_path="data/output/trocr_output.xlsx"):
        super().__init__("trocr")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.output_path = output_path

    def get_preconditions(self) -> List[str]:
        return ['cell-denoiser']

    def process(self, data: dict, config: dict) -> List[Dict]:
        denoised_pages = data['cell-denoiser']
        print(len(denoised_pages))
        print(len(denoised_pages[0]))
        trocr_results = []
        excel_data = []

        for page_idx, page in enumerate(denoised_pages):
            print(f"length of page {page_idx}: {len(page)}")
            page_result = {"columns": []}
            page_df_columns = []

            for col_idx, column in enumerate(page["columns"]):
                col_cells = []
                col_texts = []

                for cell_img in column:
                    # Convert to PIL and process with TrOCR
                    pil_image = Image.fromarray(cell_img).convert("RGB")
                    inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        generated_ids = self.model.generate(**inputs)
                        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    # Append result
                    col_cells.append({
                        "image": cell_img,
                        "erkannt": text,
                        "verbesserung": ""
                    })
                    col_texts.append(text)


                page_result["columns"].append(col_cells)
                page_df_columns.append(col_texts)

            # Build Excel dataframe
            num_rows = max(len(col) for col in page_df_columns)
            rows = []

            for i in range(num_rows):
                row = [col[i] if i < len(col) else "" for col in page_df_columns]
                rows.append(row)

            df = pd.DataFrame(rows)
            excel_data.append(df)
            trocr_results.append(page_result)

        # Save Excel
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with pd.ExcelWriter(self.output_path, engine='openpyxl') as writer:
            for i, df in enumerate(excel_data):
                df.to_excel(writer, sheet_name=f"Page_{i+1}", index=False)

        return trocr_results
