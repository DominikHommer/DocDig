from .module_base import Module
from typing import List, Dict
import os
import json
from rapidfuzz import process, fuzz

class FuzzyMatching(Module):
    def __init__(self, class_label_path: str = "./config/class_indices.json", score_threshold: int = 10):
        super().__init__("fuzzy-corrector")
        self.class_labels = []
        self.score_threshold = score_threshold

        if os.path.exists(class_label_path):
            with open(class_label_path, "r", encoding="utf-8") as f:
                name_to_idx = json.load(f)
                # Extract just the class names
                self.class_labels = list(name_to_idx.keys())

    def get_preconditions(self) -> List[str]:
        # Accepts output from any detector module with same structure
        return ['trocr', 'predictor', 'predictor-dummy']

    def process(self, data: dict, config: dict) -> List[Dict]:

        valid_keys = self.get_preconditions()
        input_key = next((k for k in data if k in valid_keys), None)
        if input_key is None:
            raise ValueError("No valid input found for fuzzy matching.")

        pages = data[input_key]
        if isinstance(pages, dict):
            pages = [pages]

        output = []

        for page in pages:
            processed_page = {"columns": []}

            for column in page["columns"]:
                is_spezies_spalte = column.get("is_spezies_spalte", False)
                cells = column.get("cells", [])
                processed_cells = []

                if not is_spezies_spalte:
                    # Spalte nicht bearbeiten, einfach übernehmen
                    processed_page["columns"].append(column)
                    continue

                for cell in cells:
                    if cell.get("skip_ocr", False):
                        print("Skipped due to questionmark fuzzy")
                        processed_cells.append(cell)
                    else:
                        text = cell.get("erkannt", "")
                        best_match, score, _ = process.extractOne(
                            text,
                            self.class_labels,
                            scorer=fuzz.token_sort_ratio
                        ) if self.class_labels else ("", 0, None)

                        correction = best_match if score >= self.score_threshold else ""
                        cell["erkannt"] = correction
                        cell["score"] = score

                        processed_cells.append(cell)

                processed_page["columns"].append({
                    "cells": processed_cells,
                    "is_spezies_spalte": True
                })

            output.append(processed_page)
    
        return output
