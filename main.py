from segmentation_client import SegmentationClient

import pandas as pd
import numpy as np
import math
import cv2 as cv
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv 
import os

def load_and_segment_pdf(pdf_path, column_numbers, main_dir):
    sgc = SegmentationClient(main_dir)

    # Schritt 1: PDF laden und in Zellen segmentieren
    table = []
    for col_nr in column_numbers:
        table.append(sgc.pdf_scan_to_cells_of_columns(pdf_path, col_nr))
    return table

def process_images(table, bird_cnn, number_cnn):
    # Schritt 2: Zellen für Texterkennung und Zahlenklassifikation vorbereiten
    csvSpecies, csvNr = [], []
    for idx, extract in enumerate(table):
        for pNr, page in enumerate(extract):
            for rNr, row in enumerate(page):
                if idx == 0:  # Texterkennung (Vogelarten)
                    species_result = classify_species(row, bird_cnn)
                    csvSpecies.append([pNr, rNr, species_result])
                elif idx == 1:  # Zahlenklassifizierung
                    number_result = classify_numbers(row, number_cnn)
                    csvNr.append([pNr, rNr, number_result])
    return csvSpecies, csvNr

def prepare_species_image_for_cnn(image):
    data = cv.bitwise_not(image)

    # TODO: Resize image canvas to correct aspect ratio and center content

    cv.imwrite("tmp.png", image)

    species_img = image.load_img("tmp.png", target_size=(22, 150))
    species_img_array = image.img_to_array(species_img)
    species_img_array = np.expand_dims(species_img_array, axis=0)
    species_img_array /= 255.0
    return species_img_array

def get_word_of_species_index(predicted_species, globalSpeciesJsonPath ='/Users/MeinNotebook/Google Drive/Meine Ablage/Scans/class_indices.json' ):
    with open(globalSpeciesJsonPath, 'r') as file:
        class_indices = json.load(file)
    species_index_to_class = {v: k for k, v in class_indices.items()}
    return species_index_to_class[predicted_species]

def classify_species(image, bird_cnn):
    # Texterkennung (Vogelarten) mit CNN
    species_img = prepare_species_image_for_cnn(image)
    species_prediction = bird_cnn.predict(species_img)
    predicted_species = np.argmax(species_prediction, axis=1)[0]
    predicted_species_in_word = get_word_of_species_index(predicted_species)
    return predicted_species_in_word

def remove_noise_increas_increase_contrast(data):
    componentsNumber, labeledImage, componentStats, componentCentroids = cv.connectedComponentsWithStats(data, connectivity=4)
    minArea = 40
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]
    data = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')
    return data

def fix_holes_between_lines_of_digits(data):
    maxKernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    data = cv.morphologyEx(data, cv.MORPH_CLOSE, maxKernel, None, None, 1, cv.BORDER_REFLECT101)
    rowCopy = np.copy(data)
    rowCopy = cv.cvtColor(rowCopy,cv.COLOR_GRAY2RGB)
    return rowCopy

def find_contour_bounding_boxes_cut_out_digits_and_predict(data, number_cnn):
    contours, hierarchy = cv.findContours(data, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = []

    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv.approxPolyDP(c, 3, True)
            boundRect.append(cv.boundingRect(contours_poly[i]))
    
    boundRect.sort(key=lambda x: x[0])

    ziffer = ""
    for i in range(len(boundRect)):
        x, y, w, h = boundRect[i]

        # Try to remove detected noise bounding boxes, which are definitely no digits
        if w > h * 2:
            continue

        if h < 10:
            continue

        if w < 10:
            if h < 20:
                continue

        croppedImg = data[y:y + h, x:x + w]

        # Make cropped image rectangular and center it
        if w > h:
            canvas = np.zeros((w, w))

            offset = math.floor((w - h) / 2)
            offsetEnd = offset
            if offsetEnd * 2 < (w - h):
                offsetEnd = offsetEnd + 1

            canvas[offset:-offsetEnd, :] = croppedImg
            croppedImg = canvas
            
        elif h > w:
            canvas = np.zeros((h, h))

            offset = math.floor((h - w) / 2)
            offsetEnd = offset
            if offsetEnd * 2 < (h - w):
                offsetEnd = offsetEnd + 1

            canvas[:, offset:-offsetEnd] = croppedImg
            croppedImg = canvas
            
        croppedImg = cv.resize(croppedImg, dsize=(28, 28), interpolation=cv.INTER_CUBIC)
        cv.imshow(croppedImg)
        croppedImg = np.expand_dims(croppedImg, axis=0)

        prediction = number_cnn.predict(croppedImg)
        predicted_digit = np.argmax(prediction)
        ziffer = ziffer + str(predicted_digit)
        
    predictedNumber = ziffer
    print("Predicted: ", predictedNumber)
    return predictedNumber

def prepare_number_image_for_cnn(image, number_cnn):
    data = cv.bitwise_not(image)
    data = remove_noise_increas_increase_contrast(data)
    #rowCopy = fix_holes_between_lines_of_digits(data)
    number = find_contour_bounding_boxes_cut_out_digits_and_predict(data, number_cnn)   # unbedingt ändern haha
    return number

def classify_numbers(image, number_cnn):
    # Zahlenerkennung mit CNN (MNIST)
    predicted_number = prepare_number_image_for_cnn(image, number_cnn)
    return predicted_number

def save_results(csvSpecies, csvNr, species_output, numbers_output):
    # Ergebnisse speichern
    pd.DataFrame(csvSpecies, columns=["Seite", "Zeile", "Vogelart"]).to_csv(species_output, index=False)
    pd.DataFrame(csvNr, columns=["Seite", "Zeile", "Zahl"]).to_csv(numbers_output, index=False)

def main():
    main_dir = os.getenv('MAIN_DIR', '/Users/MeinNotebook/Google Drive/Meine Ablage/Scans')
    test_file = main_dir + "/1972/scan_1972_CdB_2_20231125160645.pdf"
    #MAIN_DIRECTORY = "/content/drive/MyDrive/Scans"
    #TEST_FILE = MAIN_DIRECTORY + "/1972/scan_1972_CdB_2_20231125160645.pdf"
    
    #species_model_path = '/content/drive/MyDrive/Scans/Models/vogelarten_best_model.keras'
    #number_model_path = '/content/drive/MyDrive/Scans/Models/mnist_cnn_model.keras'
    #class_indices_path = '/content/drive/MyDrive/Scans/class_indices.json'
    #species_output = '/content/drive/MyDrive/Scans/predictions.csv'
    #numbers_output = '/content/drive/MyDrive/Scans/predictions_numbers.csv'
    
    species_model_path = os.getenv('SPECIES_KERAS_DIR', 'vogelarten_best_model.keras')
    number_model_path = os.getenv('MNIST_KERAS_DIR', 'mnist_cnn_model.keras')

    pdf_path = test_file

    bird_cnn = load_model(species_model_path)
    number_cnn = load_model(number_model_path)
    species_output = os.getenv('SPECIES_OUTPUT_DIR', '/Users/MeinNotebook/Desktop/predictions.csv')
    numbers_output = os.getenv('NUMBERS_OUTPUT_DIR', '/Users/MeinNotebook/Desktop/predicitons_numbers.csv')
    
    table = load_and_segment_pdf(pdf_path, [1, 6], main_dir)
    csvSpecies, csvNr = process_images(table, bird_cnn, number_cnn)
    save_results(csvSpecies, csvNr, species_output, numbers_output)

if __name__ == "__main__":
    load_dotenv()
    main()
