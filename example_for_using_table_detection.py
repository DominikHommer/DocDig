import os
import math
from PIL import Image
from table_transformer.src.inference import TableExtractionPipeline


class TableDetector:
    """
    Minimalistische Klasse, die mit der TableExtractionPipeline ausschließlich
    die Tabellen in einem Bild erkennt (Detect) und diese Bilder als Crops speichert.
    """

    def __init__(
        self,
        det_config_path: str,
        det_model_path: str,
        str_config_path: str,
        str_model_path: str,
        device: str = "cpu",
        crops_folder: str = "crops",
        padding: int = 10
    ):
        """
        :param det_config_path: Pfad zur JSON-Config für die Table Detection
        :param det_model_path: Pfad zum DETR-Checkpoint für die Table Detection
        :param str_config_path: Pfad zur JSON-Config für die Struktur-Erkennung (wird hier nur benötigt, 
                                weil die Pipeline es voraussetzt, kann aber ungenutzt bleiben)
        :param str_model_path: Pfad zum DETR-Checkpoint für die Struktur-Erkennung (dito)
        :param device: "cpu" (default), "cuda" oder "mps" (Apple Silicon)
        :param crops_folder: Wohin die ausgeschnittenen Tabellen gespeichert werden
        :param padding: Zusätzlicher Rand in Pixeln um die erkannte Tabelle
        """

        self.det_config_path = det_config_path
        self.det_model_path = det_model_path
        self.str_config_path = str_config_path
        self.str_model_path = str_model_path
        self.device = device
        self.crops_folder = crops_folder
        self.padding = padding
        
        self.pipe = TableExtractionPipeline(
            det_config_path=self.det_config_path,
            det_model_path=self.det_model_path,
            det_device=self.device,
            str_config_path=self.str_config_path,
            str_model_path=self.str_model_path,
            str_device=self.device
        )
        os.makedirs(self.crops_folder, exist_ok=True)

    def detect_tables(self, img_path: str):
        """
        Führt die Table-Detection aus, schneidet jede gefundene Tabelle
        mit Padding aus und speichert sie im crops_folder.

        :param img_path: Pfad zum Originalbild (z.B. "/path/to/1.jpg")
        :return: Liste mit Pfaden zu den gespeicherten Crops
        """
        img = Image.open(img_path).convert("RGB")

        detections = self.pipe.detect(
            img=img,
            tokens=None,      
            out_objects=True 
        )

        obj_list = detections.get("objects", [])

        saved_files = []
        for i, obj in enumerate(obj_list, start=1):
            label = obj["label"]
            score = obj["score"]
            (xmin, ymin, xmax, ymax) = obj["bbox"]

            print(f"[INFO] Objekt #{i}: Label={label}, Score={score:.3f}, BBox={obj['bbox']}")

            left   = max(0, math.floor(xmin) - self.padding)
            top    = max(0, math.floor(ymin) - self.padding)
            right  = min(img.width, math.ceil(xmax) + self.padding)
            bottom = min(img.height, math.ceil(ymax) + self.padding)

            cropped_img = img.crop((left, top, right, bottom))
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            crop_filename = f"{base_name}_table_{i}.jpg"
            crop_path = os.path.join(self.crops_folder, crop_filename)
            cropped_img.save(crop_path)
            print(f"[INFO] → Gespeichert als {crop_path}")
            saved_files.append(crop_path)

        return saved_files


detector = TableDetector(
    det_config_path="detection_config.json",
    det_model_path="pubtables1m_detection_detr_r18.pth",
    str_config_path="structure_config.json",
    str_model_path="pubtables1m_structure_detr_r18.pth",
    device="cpu",     ### keine Garantie für gpu     
    crops_folder="cut_out_table_example",   
    padding=10            
)

img_path = "rotated_tables/rotated_img_1.jpg" ### Funktioniert nur mit .jpg
saved_crops = detector.detect_tables(img_path)

print("Crops gespeichert:", saved_crops)


