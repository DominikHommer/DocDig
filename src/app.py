import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
from PIL import Image
import random
import base64

from pipeline.cv_pipeline import CVPipeline
from modules.pdf_converter import PdfConverter
from modules.tatr_extraction import TatrExtractor
from modules.table_rotator import TableRotator
from modules.column_extractor import ColumnExtractor
from modules.row_extractor import RowExtractor
from modules.detect_columns import DetectColumns
from modules.reorder_columns import ReorderColumns
from modules.cell_denoiser import CellDenoiser
from modules.cell_formatter import CellFormatter
from modules.quotation_mark_detector import QuotationMarkDetector
from modules.trocr import TrOCR
from modules.fuzzy_matching import FuzzyMatchingBirdNames
from modules.predictor_dummy import PredictorDummy



def get_base64_image(path):
    with open(path, "rb") as img_file:
        data = img_file.read()
    return base64.b64encode(data).decode()

image_base64 = get_base64_image("src/background.png")

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .stTextInput > div > div > input {
        padding-top: 6px;
        padding-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
    .custom-header {{
        background: linear-gradient(to bottom, rgba(255,255,255,0.0), rgba(255,255,255,1)),
                    url("data:image/jpg;base64,{image_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: top center;
        height: 600px;
        width: 100%;
    }}
    </style>

    <div class="custom-header"></div>
    """,
    unsafe_allow_html=True
)

uploaded = st.file_uploader(label="File upload", type="pdf", label_visibility="hidden")
if not uploaded:
    st.stop()

input_dir = os.path.join("data", "input")
os.makedirs(input_dir, exist_ok=True)
pdf_path = os.path.join(input_dir, uploaded.name)
with open(pdf_path, "wb") as f:
    f.write(uploaded.getbuffer())


if "all_pages" not in st.session_state:
    base_pipeline = CVPipeline(input_data={})
    base_pipeline.add_stage(PdfConverter(debug=False))
    st.session_state.all_pages = base_pipeline.run(input_data=pdf_path)
    st.session_state.predictions = [None] * len(st.session_state.all_pages)
    st.session_state.page_idx = 0
    st.session_state.processing = [False] * len(st.session_state.all_pages)

idx = st.session_state.page_idx
all_pages = st.session_state.all_pages
page_path = all_pages[idx]

if st.session_state.predictions[idx] is None and not st.session_state.processing[idx]:

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(
        page_path,
        caption=f"Original Seite {idx+1}/{len(all_pages)}",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Seite verarbeiten", use_container_width=True):
        st.session_state.processing[idx] = True
        st.rerun()

elif st.session_state.processing[idx]:

    st.markdown(
        "<div style='text-align:center; font-size: 24px;'>⏳ Seite wird verarbeitet...</div>",
        unsafe_allow_html=True,
    )

    page_input = {"pdf-converter": [page_path]}
    page_pipeline = CVPipeline(input_data=page_input)
    page_pipeline.add_stage(TableRotator(debug=False))
    page_pipeline.add_stage(TatrExtractor(debug=False))
    page_pipeline.add_stage(ColumnExtractor(debug=False))
    page_pipeline.add_stage(RowExtractor(debug=False))
    page_pipeline.add_stage(DetectColumns())
    page_pipeline.add_stage(ReorderColumns())
    page_pipeline.add_stage(CellDenoiser(debug=True))

    page_pipeline.add_stage(CellFormatter())
    page_pipeline.add_stage(QuotationMarkDetector())
    page_pipeline.add_stage(TrOCR())
    page_pipeline.add_stage(FuzzyMatchingBirdNames())

    result = page_pipeline.run()
    #print(f"ResultsLen: {len(result)}\nResultsLen result['columns'] {len(result['columns'])}")
    


    st.session_state.predictions[idx] = result
    st.session_state.processing[idx] = False
    st.rerun()

else:
    pred = st.session_state.predictions[idx]

    # Unpack single-page result if needed
    if isinstance(pred, list) and len(pred) == 1:
        pred = pred[0]

    columns = pred["columns"]
    cells_only = [col["cells"] for col in columns]
    rows = list(zip(*cells_only))

    st.markdown("## 🧾 Erkannte Zellstruktur")

    # ---- HEADER (row 0) ----
    header_row = rows[0]
    header_cols = st.columns(len(header_row))
    for col_idx, cell in enumerate(header_row):
        with header_cols[col_idx]:
            img_array = cell["image"]
            if img_array is None:
                st.markdown('<span style="color:red">Cell Not Detected</span>', unsafe_allow_html=True)
            else:
                img_uint8 = (
                    (img_array * 255).astype("uint8")
                    if img_array.max() <= 1
                    else img_array.astype("uint8")
                )
                st.image(Image.fromarray(img_uint8), use_container_width=True)

    # ---- OTHER ROWS (starting from row 1) ----
    for row_idx, row in enumerate(rows[1:], start=1):

        # Display Images
        image_cols = st.columns(len(row))
        for col_idx, cell in enumerate(row):
            with image_cols[col_idx]:
                img_array = cell["image"]
                if img_array is None:
                    st.markdown('<span style="color:red">Cell Not Detected</span>', unsafe_allow_html=True)
                else:
                    img_uint8 = (
                        (img_array * 255).astype("uint8")
                        if img_array.max() <= 1
                        else img_array.astype("uint8")
                    )
                    st.image(Image.fromarray(img_uint8), use_container_width=True)

        erkannt_cols = st.columns(len(row))

        # Display recognized text
        for col_idx, cell in enumerate(row):
            with erkannt_cols[col_idx]:
                erkannt_text = cell.get("erkannt", "")
                score = cell.get("score", -1)

                if score >= 90:
                    color = "#d4edda"  # green
                elif score >= 50:
                    color = "#fff3cd"  # yellow
                else:
                    color = "#f8d7da"  # red

                st.markdown(
                    f"""
                    <div style='background-color:{color}; color: black; padding:6px; border-radius:6px; text-align:center;'>
                        <b>Erkannt:</b> {erkannt_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Input field for correction
        input_cols = st.columns(len(row))
        for col_idx, _ in enumerate(row):
            with input_cols[col_idx]:
                cell_key = f"verbesserung-{idx}-{row_idx}-{col_idx}"
                st.text_input(
                    "Verbesserung:",
                    key=cell_key,
                    label_visibility="collapsed",
                    placeholder="Edit",
                )

    print(f"\nDone printing!\n")

    if st.session_state.page_idx + 1 < len(all_pages):
        if st.button("➡️ Nächste Seite"):
            st.session_state.page_idx += 1
            st.rerun()
    else:
        st.success("Alle Seiten verarbeitet 🎉")
