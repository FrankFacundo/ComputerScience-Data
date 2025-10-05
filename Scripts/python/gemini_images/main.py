from google import genai
from PIL import Image
from io import BytesIO
import os

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# prompt = "Remove the watermark"
# prompt = "Remove ONLY the watermark"
# prompt = "remove the watermark, do not modify the image"
# prompt = "remove the watermark lumbreras"
# prompt = "remove the watermark lumbreras. Do not change numbers or other information."
# prompt = "Remove the watermark. Do not change numbers or other information."
prompt = "Remove the watermark from this image and restore the background so it looks clean and uniform. Keep all the original text, numbers, symbols, and map details unchanged. Ensure the image remains sharp and clear without losing any of the original information."

# image = Image.open('img_9.jpeg')
image = Image.open('img.png')

response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[prompt, image],
)

for part in response.candidates[0].content.parts:
  if part.text is not None:
    print(part.text)
  elif part.inline_data is not None:
    image = Image.open(BytesIO(part.inline_data.data))   
    image.save("generated_image.png")
