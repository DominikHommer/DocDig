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
## Uncomment if extracted table structure images already exists
#input_data = {
#    'tatr-extractor': [
#        'data/input/tatr/page_0.jpg',
#        'data/input/tatr/page_1.jpg',
#        'data/input/tatr/page_2.jpg',
#        'data/input/tatr/page_3.jpg',
#        'data/input/tatr/page_4.jpg',
#        'data/input/tatr/page_5.jpg',
#        'data/input/tatr/page_6.jpg',
#        'data/input/tatr/page_7.jpg',
#        'data/input/tatr/page_8.jpg',
#        'data/input/tatr/page_9.jpg',
#        'data/input/tatr/page_10.jpg',
#        'data/input/tatr/page_11.jpg',
#        'data/input/tatr/page_12.jpg',
#        'data/input/tatr/page_13.jpg',
#        'data/input/tatr/page_14.jpg',
#        'data/input/tatr/page_15.jpg',
#        'data/input/tatr/page_16.jpg',
#        'data/input/tatr/page_17.jpg',
#        'data/input/tatr/page_18.jpg',
#    ]
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