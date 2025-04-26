import os
import cv2
import numpy as np
import shutil
import math
import statistics

from libs.cv_helpers import getVerticalLines
from .module_base import Module

class ColumnExtractorResult:
    columns_rgb: list[np.ndarray]
    columns_gray: list[np.ndarray]
    split_widths: list[float]

class ColumnExtractor(Module):
    def __init__(self,
                 debug: bool = False,
                 debug_folder: str = "debug/debug_column_extractor/"):
        super().__init__("column-extractor")
        self.debug = debug
        self.debug_folder = debug_folder

        self.xThres = 40
        self.minFoundLines = 2

        if self.debug:
            if os.path.exists(self.debug_folder):
                shutil.rmtree(self.debug_folder)

            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['tatr-extractor']

    def process(self, data: dict, config: dict) -> list[str]:
        file_paths: list[str] = data['tatr-extractor']

        result: list[ColumnExtractorResult] = []
        for page_i, path in enumerate(file_paths):
            base_img = cv2.imread(path, cv2.IMREAD_COLOR)
            gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            gray = cv2.bitwise_not(gray_img)

            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            iHeight, iWidth = thresh.shape[:2]
            iHeightPart = math.floor(iHeight / 8)
            if iHeightPart % 2 == 0:
                iHeightPart = iHeightPart + 1

            blur = cv2.GaussianBlur(thresh, (5, iHeightPart), 0)
            blur = cv2.GaussianBlur(blur, (3, iHeightPart), 0)
            blur = cv2.GaussianBlur(blur, (1, iHeightPart), 0)

            threshRGB = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
            threshCopy = np.copy(threshRGB)

            verticalLines = getVerticalLines(blur, self.xThres, self.minFoundLines)

            # Skip image if no vertical lines found
            if not verticalLines:
                return
            
            xSplits = [[0, 0]]

            for i in range(len(verticalLines)):
                vLine = verticalLines[i]
                x1, y1, x2, y2 = vLine[0]

                # This basically gets the maximum (possible) size of the column
                start = 0
                if (i > 0):
                    pLine = verticalLines[i - 1]

                    px1 = pLine[0][0]
                    px2 = pLine[0][2]

                    start = px1
                    if (px2 < px1):
                        start = px2

                end = x1
                if (x2 > end):
                    end = x2

                xSplits.append([start, end])
                if self.debug:
                    cv2.line(threshCopy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.debug:
                cv2.imwrite(f"{os.path.dirname(self.debug_folder)}/page_{page_i}_columns.jpg", threshCopy)

            xSplits.append([xSplits[-1][-1], iWidth])
            xSplits = np.sort(xSplits, axis=0)

            doneSplits = []
            doneSplitsRGB = []
            splitWidths = []
            for splits in xSplits:
                start, end = splits

                if start == 0:
                    continue

                if end - start <= self.xThres:
                    continue

                splitWidths.append(end - start)
                doneSplits.append(thresh[:, start:end + 15])
                doneSplitsRGB.append(base_img[:, start:end + 15])

            page: ColumnExtractorResult = {
                'columns_rgb': doneSplitsRGB,
                'columns_gray': doneSplits,
                'split_widths': splitWidths,
            }

            result.append(page)

            if self.debug:
                for j, col_img in enumerate(doneSplitsRGB):
                    cv2.imwrite(f"{os.path.dirname(self.debug_folder)}/page_{page_i}_column_{j}.jpg", col_img)

        # Post-Processing step
        # We somehow need to unify column lengths
        lenArr = []
        for _, res in enumerate(result):    
            lenArr.append(len(res['columns_rgb']))

        n_results = len(result)
        max_len   = max(len(r["split_widths"]) for r in result)
        data = np.full((n_results, max_len), np.nan)
        for i, r in enumerate(result):
            w = r["split_widths"]
            data[i, :len(w)] = w

        mean_widths = np.nanmean(data, axis=0)

        # We assume the median is the correct amount
        lenMedian = statistics.median(lenArr)
    
        continue_from = None
        formatted_result = []
        for page in result:
            maxLen = len(page['columns_gray'])
            allowed_spread = abs(maxLen - lenMedian)

            if allowed_spread == 0:
                formatted_result.append(page)

                continue
            
            c_rgb = []
            c_gray = []
            c_widths = []
            for col in range(lenMedian):
                selected_col_nr = None
                for spread in range(col, col + allowed_spread + 1):
                    if spread >= maxLen:
                        continue

                    print("spread", spread, col)
                    print(abs(page['split_widths'][spread] - mean_widths[col]))
                    print(mean_widths[col] / 2)
                    if not selected_col_nr and abs(page['split_widths'][spread] - mean_widths[col]) <= (mean_widths[col] / 2):
                        selected_col_nr = spread
                        print(selected_col_nr)

                if selected_col_nr is None:
                    c_rgb.append([])
                    c_gray.append([])
                    c_widths.append([])

                    continue
                
                if col != selected_col_nr:
                    allowed_spread -= 1

                c_rgb.append(page['columns_rgb'][selected_col_nr]),
                c_gray.append(page['columns_gray'][selected_col_nr])
                c_widths.append(page['split_widths'][selected_col_nr])

                # All spreads consumed, fill table as it is now
                if allowed_spread == 0:
                    continue_from = selected_col_nr
                    break

            if continue_from is not None:
                for col in range(continue_from, lenMedian):
                    if len(page['columns_rgb']) >= col:
                        c_rgb.append([])
                        c_gray.append([])
                        c_widths.append([])
                    else:
                        c_rgb.append(page['columns_rgb'][col]),
                        c_gray.append(page['columns_gray'][col])
                        c_widths.append(page['split_widths'][col])

            page = {
                "columns_rgb": c_rgb,
                "columns_gray": c_gray,
                "split_widths": c_widths,
            }

            formatted_result.append(page)

        return formatted_result

