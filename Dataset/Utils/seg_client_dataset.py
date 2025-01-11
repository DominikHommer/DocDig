import math
import cv2 as cv
import numpy as np
import statistics

class SegmentationClientDataSet:
    def __init__(self, path='/Users/MeinNotebook/Google Drive/Meine Ablage/Scans') -> None:
        self.MAIN_DIRECTORY = path

    def getVerticalLines(self, blur, xThres=40, minFoundLines=3, minLineLength=50):
        denoised = cv.fastNlMeansDenoising(blur, None, 30)
        dst_img = cv.Canny(denoised, 10, 200)

        lines = cv.HoughLinesP(
            dst_img,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=minLineLength,
            maxLineGap=2
        )
        if lines is None:
            return []

        verticalLines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if not ((x1 + xThres) > x2 and (x1 - xThres) < x2):
                continue

            shouldAdd = True
            for i in range(len(verticalLines)):
                vLine = verticalLines[i]
                v_x1, v_y1, v_x2, v_y2 = vLine[0]

                if (y1 < v_y1) and ((v_x1 + xThres) > x1 and (v_x1 - xThres) < x1):
                    verticalLines[i] = [[x1, y1, v_x2, v_y2], vLine[1] + 1]
                    shouldAdd = False
                elif (y2 > v_y2) and ((v_x2 + xThres) > x2 and (v_x2 - xThres) < x2):
                    verticalLines[i] = [[v_x1, v_y1, x2, y2], vLine[1] + 1]
                    shouldAdd = False

            if shouldAdd:
                verticalLines.append([line[0], 0])

        return list(filter(lambda l: l[1] >= minFoundLines, verticalLines))

    def calculate_center_of_table(self, gray_img):
        _, binary = cv.threshold(gray_img, 150, 255, cv.THRESH_BINARY_INV)

        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = gray_img.shape[:2]
            return (w // 2, h // 2)

        contour = max(contours, key=cv.contourArea)
        (h, w) = gray_img.shape[:2]
        default_center = (w // 2, h // 2)

        global_min_y = min(pt[0][1] for pt in contour)
        global_max_y = max(pt[0][1] for pt in contour)
        middle_y = (global_min_y + global_max_y) / 2

        upper_half = [pt[0] for pt in contour if pt[0][1] < middle_y]
        lower_half = [pt[0] for pt in contour if pt[0][1] >= middle_y]

        if not upper_half or not lower_half:
            return default_center

        top_left = (min(pt[0] for pt in upper_half), min(pt[1] for pt in upper_half))
        top_right = (max(pt[0] for pt in upper_half), min(pt[1] for pt in upper_half))
        bottom_left = (min(pt[0] for pt in lower_half), max(pt[1] for pt in lower_half))
        bottom_right = (max(pt[0] for pt in lower_half), max(pt[1] for pt in lower_half))

        center_x = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) // 4
        center_y = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) // 4
        center = (center_x, center_y)

        allowed_deviation = 100
        if (
            abs(center[0] - default_center[0]) > allowed_deviation or
            abs(center[1] - default_center[1]) > allowed_deviation
        ):
            return default_center
        return center

    def _compute_rotation_angle(self, gray_img, vertical_lines):
        vertical_lines.sort(key=lambda x: x[0][0])
        filteredVerticalLines = []
        if vertical_lines:
            filteredVerticalLines.append(vertical_lines[0])
            
        for i in range(1, len(vertical_lines)):
            current_line = vertical_lines[i]
            prev_line = vertical_lines[i - 1]
            distance = current_line[0][0] - prev_line[0][0]
            if distance > 120:
                filteredVerticalLines.append(current_line)
                
        angles_to_horizontal = []
        for line in filteredVerticalLines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx != 0:
                slope = (y2 - y1) / dx
                angle_deg = np.degrees(np.arctan(slope))
                angles_to_horizontal.append(angle_deg)

        if not angles_to_horizontal:
            return 0.0, (gray_img.shape[1] // 2, gray_img.shape[0] // 2)

        median_angle = np.median(angles_to_horizontal)
        filtered_angles = [
            angle for angle in angles_to_horizontal
            if abs(angle - median_angle) < 0.5
        ]
        if not filtered_angles:
            return 0.0, (gray_img.shape[1] // 2, gray_img.shape[0] // 2)

        rotation_angle = np.mean(filtered_angles)
        if rotation_angle < -45:
            rotation_angle += 90
        elif rotation_angle > 45:
            rotation_angle -= 90
        center = self.calculate_center_of_table(gray_img)

        return rotation_angle, center

    def process_and_rotate_color(self, imgPath, xThres=40, minFoundLines=3):
        color_img = cv.imread(imgPath, cv.IMREAD_COLOR)
        assert color_img is not None, f"Bild konnte nicht geladen werden: {imgPath}"

        gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

        gray_inverted = cv.bitwise_not(gray_img)
        thresh = cv.threshold(gray_inverted, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        blur = cv.GaussianBlur(thresh, (1, 31), 0)
        blur = cv.GaussianBlur(blur, (3, 1), 0)
        blur = cv.GaussianBlur(blur, (3, 21), 0)
        blur = cv.GaussianBlur(blur, (3, 31), 0)
        blur = cv.GaussianBlur(blur, (3, 1), 0)

        vertical_lines = self.getVerticalLines(
            blur,
            xThres=xThres,
            minFoundLines=minFoundLines
        )

        if not vertical_lines:
            print(f"Keine vertikalen Linien erkannt (Bild: {imgPath}) – kein Rotieren.")
            return color_img

        angle, center = self._compute_rotation_angle(gray_img, vertical_lines)

        cx, cy = center 
        M = cv.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
        (h, w) = color_img.shape[:2]

        rotated_color = cv.warpAffine(
            color_img,
            M,
            (w, h),
            flags=cv.INTER_CUBIC,
            borderMode=cv.BORDER_REPLICATE
        )
        return rotated_color
    
    def getYStartEndForLine(self, i, lines):
        hLine = lines[i]
        _, y1, _, y2 = hLine[0]

        # This basically gets the maximum (possible) height of the row
        start = 0
        if (i > 0):
            pLine = lines[i - 1]

            py1 = pLine[0][1]
            py2 = pLine[0][3]

            start = py1
            if (py2 < py1):
                start = py2

        end = y1
        if (y2 > end):
            end = y2

        return start, end
    
    def imageToCells(self, imgPath, colNr, useCellHeightMedian = False):
        img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"

        xThres = 40
        minFoundLines = 3

        # Detect vertical lines of image and try to rotate it
        gray = cv.bitwise_not(img)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        iHeight, iWidth = thresh.shape[:2]
        iHeightPart = math.floor(iHeight / 8)
        if iHeightPart % 2 == 0:
          iHeightPart = iHeightPart + 1



        xThres = 40
        minFoundLines = 2

        gray = cv.bitwise_not(img)

        #cv.imwrite(imgPath + "-gray.jpg", cv.cvtColor(gray ,cv.COLOR_GRAY2RGB))
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        blur = cv.GaussianBlur(thresh, (5, iHeightPart), 0)
        blur = cv.GaussianBlur(blur, (3, iHeightPart), 0)
        blur = cv.GaussianBlur(blur, (1, iHeightPart), 0)

        threshRGB = cv.cvtColor(blur,cv.COLOR_GRAY2RGB)
        threshCopy = np.copy(threshRGB)

        verticalLines = self.getVerticalLines(blur, xThres, minFoundLines)
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
            cv.line(threshCopy,(x1,y1),(x2,y2),(0,255,0),2)

        #cv2_imshow(threshCopy)
        #cv.imwrite(imgPath + "-vertical.jpg", threshCopy)

        xSplits.append([xSplits[-1][-1], iWidth])
        xSplits = np.sort(xSplits, axis=0)

        doneSplits = []
        for splits in xSplits:
            start, end = splits

            if start == 0:
                continue

            if end - start <= xThres:
                continue

            doneSplits.append(gray[:, start:end + 15])

        # Skip image if not enough columns are detected
        if len(doneSplits) < 7:
            print(f"Not enough columns detected: {len(doneSplits)}")
            return

        # Check if the column index is within bounds
        if colNr >= len(doneSplits):
            print(f"Column index {colNr} out of range. Total columns: {len(doneSplits)}.")
            return

        colTest = doneSplits[colNr]
        iHeight, iWidth = colTest.shape[:2]

        # Make width uneven so we can use it as kernel for blurring an image
        if iWidth % 2 == 0:
          iWidth = iWidth + 1

        # TODO: remove the header cell of a column

        #cv.imwrite(imgPath + "-column-" + str(colNr) + ".jpg", colTest)
        #cv.imwrite(imgPath + "-column-" + str(colNr) + "-white.jpg", cv.cvtColor(cv.bitwise_not(colTest) ,cv.COLOR_GRAY2RGB))

        # Segement column into cells
        copyTest = np.copy(colTest)
        copyRGB = np.copy(copyTest)
        copyRGB = cv.cvtColor(copyRGB,cv.COLOR_GRAY2RGB)

        copyTest = cv.bitwise_not(copyTest)

        if iWidth % 2 == 0:
          iWidth = iWidth + 1

        iWidthPart = math.floor(iWidth / 4)
        if iWidthPart % 2 == 0:
          iWidthPart = iWidthPart + 1

        colBlur = cv.GaussianBlur(colTest, (iWidth, 5), 0)
        colBlur = cv.GaussianBlur(colTest, (iWidth, 3), 0)
        colBlur = cv.GaussianBlur(colTest, (iWidth, 1), 0)

        #cv.imwrite(imgPath + "-column-" + str(colNr) + "-horizontal-blur.jpg", colBlur)

        colDenoised = cv.fastNlMeansDenoising(colBlur, None, 30)
        colCanny = cv.Canny(colDenoised, 10, 200)

        #cv.imwrite(imgPath + "-column-" + str(colNr) + "-horizontal-blur.jpg", colBlur)

        houghThreshold = 100
        minLineLength = 35

        # Small cells should have "smaller" thresholds to detect lines
        if iWidth < 200:
          houghThreshold = 50
          minLineLength = 10

        lines = cv.HoughLinesP(colCanny, 1, np.pi / 180, threshold=houghThreshold, minLineLength=minLineLength, maxLineGap=2)

        # Skip image if no horizontal lines found
        if lines is None:
          return

        # We define a 30px y-Threshold, which "decides" if a line is indeed a horizontal detected line
        yThres = 30
        horizontalLines = []
        for line in lines:
            x1,y1,x2,y2 = line[0]

            # Skip value, as it is not a horizontal line
            if not ((y1 + yThres) > y2 and (y1 - yThres) < y2):
                continue

            shouldAdd = True
            for i in range(len(horizontalLines)):
                hLine = horizontalLines[i]
                h_x1, h_y1, h_x2, h_y2 = hLine[0]

                # Check if "line" Vector is above / under "hLine" Vector and in its y-threshold
                # Update horizontalLines List accordingly
                if (x1 < h_x1) and ((h_y1 + yThres) > y1 and (h_y1 - yThres) < y1):
                    horizontalLines[i] = [[x1, y1, h_x2, h_y2], hLine[1]+1]

                    shouldAdd = False
                elif (x2 > h_x2) and ((h_y2 + yThres) > y2 and (h_y2 - yThres) < y2):
                    horizontalLines[i] = [[h_x1, h_y1, x2, y2], hLine[1]+1]

                    shouldAdd = False

            if shouldAdd == True:
                horizontalLines.append([line[0], 0])

        minFoundLines = 1
        if iWidth < 200:
          minFoundLines = 0

        ySplits = [[0, 0]]

        horizontalLines = list(filter(lambda l: l[1] >= minFoundLines, horizontalLines))

        for i in range(len(horizontalLines)):
            hLine = horizontalLines[i]
            x1, y1, x2, y2 = hLine[0]

            # This basically gets the maximum (possible) height of the row
            start, end = self.getYStartEndForLine(i, horizontalLines)
        
            ySplits.append([start, end])
            cv.line(copyRGB,(x1,y1),(x2,y2),(0,255,0),2)

        #cv2_imshow(copyRGB)
        #cv.imwrite(imgPath + "-column-" + str(colNr) + "-horizontal.jpg", copyRGB)

        ySplits.append([ySplits[-1][-1], iHeight])
        ySplits = np.sort(ySplits, axis=0)

        if useCellHeightMedian:
          realSplits = []
          heights = []

          for splits in ySplits:
            start, end = splits

            if start == 0:
                continue

            height = end - start
            if height <= yThres:
             continue
        
            heights.append(height)

          heightMedian = statistics.median(heights)
      
          for splits in ySplits:
            start, end = splits

            if start == 0:
                continue

            height = end - start
            if height <= yThres:
                continue
        
            if height >= heightMedian * 1.75:
              while height > heightMedian * 1.75:
                  newEnd = int(start + heightMedian)

                  realSplits.append([start, newEnd])

                  start = newEnd
                  height = end - start
          
              realSplits.append([start, end])
            else:
               realSplits.append(splits)

          ySplits = realSplits

        doneRowSplits = []
        for splits in ySplits:
            start, end = splits

            if start == 0:
                continue

            if end - start <= yThres:
                continue

            doneRowSplits.append(copyTest[start:end + 15,:])

        return doneRowSplits