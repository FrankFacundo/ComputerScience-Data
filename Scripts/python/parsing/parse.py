import os
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
source = "SOL-SM-2025_I.pdf"

pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True
pipeline_options.ocr_options.lang = ["es"]
pipeline_options.images_scale = 2.0

converter = DocumentConverter(
    allowed_formats=[
        InputFormat.PDF,
    ],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,  # pipeline options go here.
        )
    },
)
result = converter.convert(source)

result.document.save_as_yaml(filename=Path("output.yaml"))
result.document.save_as_json(filename=Path("output.json"))
result.document.save_as_markdown(filename=Path("output.md"), image_mode="referenced")
