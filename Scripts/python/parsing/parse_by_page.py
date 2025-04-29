import io
import logging
import os
from datetime import datetime
from pathlib import Path

import polars as pl
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)

# Set the directory containing PDFs and the output Parquet file path.
source_dir = "/home/frank/Datalake/datasets/examenes_de_admision/publicas/UNMSM - Universidad Nacional Mayor de San Marcos/2025-I"
output_parquet = "results_unmsm_2025_test.parquet"

# Configure pipeline options.
pipeline_options = PdfPipelineOptions()

accelerator_options = AcceleratorOptions(
    num_threads=16, device=AcceleratorDevice.CUDA, cuda_use_flash_attention2=True
)
pipeline_options.accelerator_options = accelerator_options

pipeline_options.do_table_structure = True
pipeline_options.do_ocr = False
pipeline_options.do_code_enrichment = False
pipeline_options.do_formula_enrichment = False
pipeline_options.do_picture_classification = False

pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pipeline_options.ocr_options.lang = ["es"]

pipeline_options.images_scale = 1.0
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = False

settings.debug.visualize_layout = True
settings.debug.visualize_ocr = False
settings.debug.visualize_tables = False
settings.debug.visualize_cells = False
settings.debug.visualize_raw_layout = False

# Initialize the DocumentConverter for PDFs.
converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,  # pipeline options go here.
        )
    },
)

results = []

# First, gather all PDF file paths from the source directory.
pdf_files = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_files.append((root, file))


# Recursively iterate over all PDF files in the directory.
for root, file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
    pdf_path = os.path.join(root, file)
    try:
        conv_res = converter.convert(pdf_path)
    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")
        continue

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    filename = Path(f"output_embedded_{current_time}.md")

    # Iterate through each page of the converted PDF.
    for page in conv_res.pages:
        page_number = page.page_no + 1
        # print(f"page: {page_number}")

        # Extract the page image.
        try:
            image = page.get_image(scale=pipeline_options.images_scale)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
        except Exception as e:
            logging.error(
                f"Failed to get image for page {page_number} in {pdf_path}: {e}"
            )
            img_bytes = None

        ############
        clusters = page.predictions.layout.clusters
        scale_x = page.image.width / page.size.width
        scale_y = page.image.height / page.size.height
        bboxes = []
        labels = []
        confidences = []
        for c_tl in clusters:
            all_clusters = [c_tl, *c_tl.children]
            for c in all_clusters:
                x0, y0, x1, y1 = c.bbox.as_tuple()
                x0 *= scale_x
                x1 *= scale_x
                y0 *= scale_x
                y1 *= scale_y
                bboxes.append([x0, y0, x1, y1])
                labels.append(c.label.value)
                confidences.append(c.confidence)

        ############

        results.append(
            {
                "image": img_bytes,
                "bboxes": bboxes,
                "label": labels,
                "confidence": confidences,
                "page_no": page_number,
                "filename": file,
                "pathfile": pdf_path,
            }
        )

# Create a Polars DataFrame and save the results as a Parquet file.
df = pl.DataFrame(results)
df.write_parquet(output_parquet)

print(f"Saved results to {output_parquet}")
