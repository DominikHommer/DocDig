from Utils.seg_client_dataset import SegmentationClientDataSet

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from TableDetector.table_detector import TableDetector

import cv2 as cv
import pytesseract
import pandas as pd
from PIL import Image

def detect_tables(src_dir, dst_dir, padding):
    config = {
    "det_config_path": "detection_config.json",
    "det_model_path": "pubtables1m_detection_detr_r18.pth",
    "str_config_path": "structure_config.json",
    "str_model_path": "pubtables1m_structure_detr_r18.pth",
    "device": "cpu",
    "crops_folder": dst_dir,
    "padding": padding
    }
    detector = TableDetector(**config)
    detector.process_directory(src_dir)

def rotate_scans(src_dir, dst_dir):
    sc = SegmentationClientDataSet()

    for subfolder in os.listdir(src_dir):
        subfolder_path = os.path.join(src_dir, subfolder)
        if os.path.isdir(subfolder_path):
            dst_subfolder_path = os.path.join(dst_dir, subfolder)
            os.makedirs(dst_subfolder_path, exist_ok=True)

            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith(".png"):
                    src_file = os.path.join(subfolder_path, filename)
                    dst_file = os.path.join(dst_subfolder_path, filename)

                    print(f"Verarbeite Datei: {src_file}")
                    try:
                        rotated_color_img = sc.process_and_rotate_color(
                            src_file,
                            xThres=40,
                            minFoundLines=3
                        )
                    except AssertionError as e:
                        print(f"Fehler beim Laden: {e}")
                        continue
                    
                    success = cv.imwrite(dst_file, rotated_color_img)
                    if success:
                        print(f"Gespeichert: {dst_file}")
                    else:
                        print(f"Fehler beim Speichern: {dst_file}")
                        

def png_to_jpg_conversion(src_dir, dst_dir, jpg_quality=95):
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        dst_subfolder = os.path.join(dst_dir, rel_path)
        os.makedirs(dst_subfolder, exist_ok=True)

        for filename in files:
            if filename.lower().endswith(".png"):
                src_file = os.path.join(root, filename)
                base_name = os.path.splitext(filename)[0]
                dst_filename = base_name + ".jpg"
                dst_file = os.path.join(dst_subfolder, dst_filename)
                img = cv.imread(src_file, cv.IMREAD_COLOR)
                if img is None:
                    print(f"Fehler beim Laden: {src_file}")
                    continue
                success = cv.imwrite(dst_file, img, [int(cv.IMWRITE_JPEG_QUALITY), jpg_quality])
                if success:
                    print(f"Gespeichert: {dst_file}")
                else:
                    print(f"Fehler beim Speichern: {dst_file}")
                    

def extract_valid_column_cells(image_path, sc):
    colNr = 0 
    max_columns = 10 

    while colNr <= max_columns:
        cells = sc.imageToCells(image_path, colNr, useCellHeightMedian=True)
        
        if not cells:
            print(f"No cells found in column {colNr} for image {image_path}.")
            colNr += 1
            continue
        first_cell = cells[0]

        cropped_cell = crop_center(first_cell, width=200, height=50)
        cropped_text = pytesseract.image_to_string(cropped_cell, lang="fra").strip().lower()
        print(f"Cropped text in first cell of column {colNr}: '{cropped_text}'")
        
        if "espèce" in cropped_text:
            print(f"Column {colNr} has 'Espèce' in the cropped first cell.")
            return cells  

        full_text = pytesseract.image_to_string(first_cell, lang="fra").strip().lower()
        print(f"Full text in first cell of column {colNr}: '{full_text}'")
        
        if "espèce" in full_text:
            print(f"Column {colNr} has 'Espèce' in the full-size first cell.")
            return cells 

        if first_cell.shape[1] >= 280:
            print(f"Column {colNr} has a valid first cell width of {first_cell.shape[1]} pixels.")
            return cells 

        print(f"Column {colNr} skipped due to first cell width of {first_cell.shape[1]} pixels.")
        colNr += 1 

    print(f"No valid columns found for image {image_path} within the first {max_columns} columns.")
    return None 


def get_images_from_folder(folder_path):
    images = []
    for file_name in sorted(os.listdir(folder_path)): 
        if file_name.endswith(".jpg"):
            images.append(os.path.join(folder_path, file_name))
    return images

def get_labels_from_excel(excel_path):
    print(f"Processing Excel file: {excel_path}")
    workbook = pd.ExcelFile(excel_path)
    labels = {}
    for sheet_name in workbook.sheet_names:
        print(f"Processing sheet: {sheet_name}")
        sheet = workbook.parse(sheet_name, header=None)

        # mind zwei Spalten check
        if sheet.shape[1] > 1:
            try:
                labels[sheet_name] = sheet.iloc[:, 1].values.tolist()
            except Exception as e:
                print(f"Error in sheet '{sheet_name}' of file '{excel_path}': {e}")
                raise
        else:
            print(f"Warning: Sheet '{sheet_name}' in file '{excel_path}' has less than two columns. Skipping.")
            labels[sheet_name] = []

    return labels

def crop_center(image, width, height):
    h, w = image.shape[:2]
    x_start = max((w - width) // 2, 0)
    y_start = max((h - height) // 2, 0)
    return image[y_start:y_start + height, x_start:x_start + width]

def create_dataset_from_folder(folder_path, excel_path, output_dir, sc):
    images = get_images_from_folder(folder_path)
    labels = get_labels_from_excel(excel_path)

    for image_path in images:
        image_name = os.path.basename(image_path)
        print(f"Processing image: {image_name}")

        sheet_name = image_name.replace(".jpg", ".png") 
        if sheet_name not in labels:
            print(f"Warning: No matching sheet for image {image_name}")
            continue
        
        image_labels = labels[sheet_name]
        cells = extract_valid_column_cells(image_path, sc)
        
        if not cells :#or len(cells) < len(image_labels):
            print(f"Warning: Mismatch between cells and labels in {image_name}")
            continue

        for cell_idx, cell in enumerate(cells):
            if cell_idx == 0:
                cropped_cell = crop_center(cell, width=200, height=50)   
                text = pytesseract.image_to_string(cropped_cell, lang="fra").strip().lower()
                print(f"Text in first cell: '{text}'")

                if "espèce" in text:
                    print(f"Skipping first cell of image {image_name} due to 'Espèce'.")
                    continue 
            
            if cell_idx - 1 >= len(image_labels):
                print(f"Warning: No label for cell {cell_idx} in image {image_name}")
                break
            
            label = str(image_labels[cell_idx - 1])
            if pd.isna(label):
                print(f"Skipping cell {cell_idx} in image {image_name} due to missing label.")
                continue
            
            label_dir = os.path.join(output_dir, label)
            
            os.makedirs(label_dir, exist_ok=True)
            output_cell_path = os.path.join(label_dir, f"{os.path.basename(folder_path)}_{image_name}_cell{cell_idx}.jpg")
            cv.imwrite(output_cell_path, cell)
            cv.imwrite("cell.png", cell)
            print("label:", label)
            print(f"Saved cell {cell_idx} to {output_cell_path}")

# Hauptverarbeitungsschleife
def process_all_folders_for_dataset(src_dir, dst_dir, excel_dir):
    sc = SegmentationClientDataSet()
    for folder_name in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue 
        
        excel_path = os.path.join(excel_dir, f"{folder_name}.xlsx")
        if not os.path.exists(excel_path):
            print(f"Warning: No matching Excel file for folder {folder_name}")
            continue
        
        print(f"Processing folder: {folder_name}")
        create_dataset_from_folder(folder_path, excel_path, dst_dir, sc)


def remove_files_in_main_folder(dataset_path, valid_extensions=(".jpg", ".jpeg", ".png")):
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isfile(item_path) and item.lower().endswith(valid_extensions):
            print(f"Entferne Datei im Hauptordner: {item}")
            os.remove(item_path)

def remove_images_by_width_in_folders(dataset_path, min_width, max_width):
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            f_path = os.path.join(root, f)
            try:
                with Image.open(f_path) as img:
                    width, height = img.size
                    if width < min_width:
                        print(f"Bild {f_path} ist zu schmal (Breite={width} < {min_width}). Entferne...")
                        os.remove(f_path)
                    elif max_width is not None and width > max_width:
                        print(f"Bild {f_path} ist zu breit (Breite={width} > {max_width}). Entferne...")
                        os.remove(f_path)
                    
            except Exception as e:
                print(f"Fehler beim Öffnen von {f_path}: {e}. Entferne Datei vorsichtshalber...")
                os.remove(f_path)

def dataset_post_processing(dataset_dir, min_width, max_width):
    remove_files_in_main_folder(dataset_dir)
    remove_images_by_width_in_folders(dataset_dir, min_width, max_width)