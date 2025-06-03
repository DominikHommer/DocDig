from .module_base import Module
from typing import List, Dict
import os
import json
from rapidfuzz import process, fuzz


class FuzzyMatching(Module):
    def __init__(self, class_label_path: str = "./config/class_labels.json", score_threshold: int = 80):
        super().__init__("fuzzy-corrector")
        self.class_labels = []
        self.score_threshold = score_threshold

        if os.path.exists(class_label_path):
            with open(class_label_path, "r", encoding="utf-8") as f:
                self.class_labels = json.load(f)

    def get_preconditions(self) -> List[str]:
        # Put the model used before her
        # Todo: Run this module if any model "trocr", "predictor", "dummypredictor", ... has run before
        return ['trocr']

    def process(self, data: dict, config: dict) -> List[Dict]:
        # Dynamically get the key for the actual input module
        input_key = next(iter(data))
        pages = data[input_key]
        print(len(pages))
        print(len(pages[0]))
        corrected_output = []

        for page in pages:
            corrected_page = {"columns": []}

            for column in page["columns"]:
                corrected_column = []

                for cell in column:
                    text = cell.get("erkannt", "")
                    best_match, score, _ = process.extractOne(
                        text,
                        self.class_labels,
                        scorer=fuzz.token_sort_ratio
                    ) if self.class_labels else ("", 0, None)

                    correction = best_match if score >= self.score_threshold else ""

                    corrected_cell = {
                        "image": cell["image"],
                        "erkannt": text,
                        "verbesserung": correction
                    }

                    corrected_column.append(corrected_cell)

                corrected_page["columns"].append(corrected_column)
            corrected_output.append(corrected_page)

        return corrected_output
