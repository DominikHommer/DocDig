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

        # Pick from valid previous module outputs
        valid_keys = self.get_preconditions()
        input_key = next((k for k in valid_keys if k in data), None)
        if input_key is None:
            raise ValueError("No valid input found for fuzzy matching.")

        pages = data[input_key]
        if isinstance(pages, dict):
            pages = [pages]

        output = []

        for page in pages:
            processed_page = {"columns": []}

            for col_idx, column in enumerate(page["columns"]):  # ✅ now safe, page is a dict
                processed_column = []

                if col_idx == 0:

                    for cell in column:

                        if cell["skip_ocr"]:
                            processed_column.append(cell)

                        else: # cell["skip_ocr"] is false
                            text = cell["erkannt"]
                            best_match, score, _ = process.extractOne(text,
                                                                      self.class_labels,
                                                                      scorer=fuzz.token_sort_ratio
                                                                      ) if self.class_labels else ("", 0, None)

                            correction = best_match if score >= self.score_threshold else ""

                            cell["text"] = correction
                            cell["score"] = score

                            processed_column.append(cell)

                processed_page["columns"].append(processed_column)

            output.append(processed_page)

        return output
