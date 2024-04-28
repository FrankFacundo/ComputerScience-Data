import os
import pathlib

from llama_parse import LlamaParse

parser = LlamaParse(
    api_key=os.getenv("LP_KEY"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    num_workers=4,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="en",  # Optionally you can define a language, default=en
)

# sync
document_md = parser.load_data(
    f"{pathlib.Path(__file__).parent.resolve()}/optilife2.pdf"
)
print(document_md)
# document_md[0].text

with open("document_md.md", "w") as f:
    f.write(document_md[0].text)

parser = LlamaParse(
    api_key=os.getenv("LP_KEY"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="text",  # "markdown" and "text" are available
    num_workers=4,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="en",  # Optionally you can define a language, default=en
)

# sync
document_text = parser.load_data(
    f"{pathlib.Path(__file__).parent.resolve()}/optilife2.pdf"
)

print(document_text)
with open("document_text.txt", "w") as f:
    f.write(document_text[0].text)
