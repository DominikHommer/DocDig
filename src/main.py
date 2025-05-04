from pipeline.cv_pipeline import CVPipeline
import os

from modules.pdf_converter import PdfConverter
from modules.tatr_extraction import TatrExtractor
from modules.table_rotator import TableRotator
from modules.column_extractor import ColumnExtractor
from modules.row_extractor import RowExtractor
from modules.cell_denoiser import CellDenoiser

input_image_path = os.path.join("data", "input", "scan_1972_CdB_1_20231125160539.pdf")

input_data = {}
### Uncomment if extracted table structure images already exists
#tatr = []
#for i in range(25, 35):
#    tatr.append(f'data/input/tatr/page_{i}.jpg')
#
#input_data = {
#    'tatr-extractor': tatr
#}

pipeline = CVPipeline(input_data=input_data)

## Uncomment to convert pdf to jpgs and extract table structure
pipeline.add_stage(PdfConverter(debug=False))
pipeline.add_stage(TableRotator(debug=False))
pipeline.add_stage(TatrExtractor(debug=False))

pipeline.add_stage(ColumnExtractor(debug=True))
pipeline.add_stage(RowExtractor(debug=True))
pipeline.add_stage(CellDenoiser(debug=True))
pipeline.run(input_data=input_image_path)