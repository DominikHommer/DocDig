import os
from Utils.utils import (
    rotate_scans,
    png_to_jpg_conversion,
    detect_tables,
    process_all_folders_for_dataset,
    dataset_post_processing,
)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    base_dir_scans = os.path.join(base_dir, "Scans_for_Dataset") 
    src_dir_scans = os.path.join(base_dir_scans, "pngs")  
    dst_dir_rotated_scans = os.path.join(base_dir_scans, "rotated_pngs")  
    rotated_jpgs_dir = os.path.join(base_dir_scans, "rotated_jpgs") 
    detected_table_dir = os.path.join(base_dir_scans, "detected_tables")  
    excel_folder = os.path.join(base_dir_scans, "excel")  # Excel-Dateien
    dataset_dir = os.path.join(base_dir_scans, "Dataset")  

    os.makedirs(src_dir_scans, exist_ok=True)
    os.makedirs(dst_dir_rotated_scans, exist_ok=True)
    os.makedirs(rotated_jpgs_dir, exist_ok=True)
    os.makedirs(detected_table_dir, exist_ok=True)
    os.makedirs(excel_folder, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    print("Drehe Scans...")
    rotate_scans(src_dir=src_dir_scans, dst_dir=dst_dir_rotated_scans)

    print("Konvertiere PNGs in JPGs...")
    png_to_jpg_conversion(src_dir=dst_dir_rotated_scans, dst_dir=rotated_jpgs_dir, jpg_quality=100)

    print("Erkenne Tabellen...")
    detect_tables(src_dir=rotated_jpgs_dir, dst_dir=detected_table_dir, padding=10)

    print("Erstelle Dataset...")
    process_all_folders_for_dataset(
        src_dir=detected_table_dir, dst_dir=dataset_dir, excel_dir=excel_folder
    )

    print("Postprocessing des Datasets...")
    dataset_post_processing(dataset_dir=dataset_dir, min_width=320, max_width=400)

    print("Fertig!")