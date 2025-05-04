import os
import streamlit as st
from pipeline.cv_pipeline import CVPipeline
from modules.pdf_converter import PdfConverter
from modules.tatr_extraction import TatrExtractor
from modules.table_rotator import TableRotator
from modules.column_extractor import ColumnExtractor
from modules.row_extractor import RowExtractor
from modules.cell_denoiser import CellDenoiser

st.set_page_config(layout="wide")

# 1) PDF hochladen
uploaded = st.file_uploader("PDF hochladen", type="pdf")
if not uploaded:
    st.stop()

# 2) temporär speichern
input_dir = os.path.join("data", "input")
os.makedirs(input_dir, exist_ok=True)
pdf_path = os.path.join(input_dir, uploaded.name)
with open(pdf_path, "wb") as f:
    f.write(uploaded.getbuffer())

# 3) Einmalig: PDF → Seitenbilder
if "all_pages" not in st.session_state:
    base_pipeline = CVPipeline(input_data={})
    base_pipeline.add_stage(PdfConverter(debug=False))
    st.session_state.all_pages = base_pipeline.run(input_data=pdf_path)
    st.session_state.processed = [None] * len(st.session_state.all_pages)
    st.session_state.page_idx = 0

# Aktuelle Seite
idx = st.session_state.page_idx
all_pages = st.session_state.all_pages
total = len(all_pages)
page_path = all_pages[idx]

# 4) UI: Zwei Spalten
col1, col2 = st.columns(2)

with col1:
    st.image(
        page_path,
        caption=f"Original Seite {idx+1}/{total}",
        use_container_width=True,
    )

with col2:
    # Initialisierung für Seite
    if "just_processed" not in st.session_state:
        st.session_state.just_processed = [False] * total

    if st.session_state.processed[idx] is not None:
        st.image(
            st.session_state.processed[idx],
            caption=f"Ergebnis Seite {idx+1}/{total} ✔️",
            use_container_width=True,
        )

        if st.session_state.processed[idx] is not None:
            if idx + 1 < total:
                if st.button("Nächste Seite anzeigen"):
                    st.session_state.page_idx += 1
                    st.session_state.just_processed[idx] = False
                    st.rerun()
            else:
                st.success("Alle Seiten verarbeitet 🎉")
        else:
            st.success(f"Seite {idx+1} bereits verarbeitet ✔️")

    else:
        if st.button("Seite verarbeiten"):
            page_input = {"pdf-converter": [page_path]}
            page_pipeline = CVPipeline(input_data=page_input)
            page_pipeline.add_stage(TableRotator(debug=False))
            page_pipeline.add_stage(TatrExtractor(debug=False))
            page_pipeline.add_stage(ColumnExtractor(debug=False))
            page_pipeline.add_stage(RowExtractor(debug=False))
            page_pipeline.add_stage(CellDenoiser(debug=False))

            result_list = page_pipeline.run()
            st.session_state.processed[idx] = result_list[0]
            st.session_state.just_processed[idx] = True
            st.rerun()

