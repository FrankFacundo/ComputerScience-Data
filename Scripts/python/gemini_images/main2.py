import base64
import json
import os
from pathlib import Path

import requests

# --- Config ---
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL = "gemini-2.5-flash-image-preview"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

# Same prompt as your Code A (typo kept as-is, but "Ghibli" is the usual spelling)
prompt = "Remove the watermark from this image and restore the background so it looks clean and uniform. Keep all the original text, numbers, symbols, and map details unchanged. Ensure the image remains sharp and clear without losing any of the original information."

# Path to the input image
image_path = Path("img.png")

# Where to save the generated image
output_path = Path("generated_image.png")


def guess_mime(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def main():
    if not API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.")

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Read source image bytes
    src_bytes = image_path.read_bytes()
    mime_type = guess_mime(src_bytes)

    # Build request payload: prompt + inline image
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(src_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY,
    }

    # Call Gemini
    resp = requests.post(URL, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Basic safety/finish checks
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("No candidates returned from Gemini API.")

    cand = candidates[0]
    # If the request was blocked, content may be missing
    if "content" not in cand:
        finish_reason = cand.get("finishReason", "UNKNOWN")
        raise RuntimeError(f"Generation failed/blocked (finishReason={finish_reason}).")

    # Parse returned parts: may include text + inline image
    parts = cand["content"].get("parts", [])
    if not parts:
        raise RuntimeError("No content parts in Gemini response.")

    image_bytes = None
    returned_mime = None

    for p in parts:
        # Optional: print any textual description
        if "text" in p and p["text"]:
            print(p["text"].strip())

        # Look for inline image
        inline = p.get("inlineData")
        if inline and inline.get("data"):
            try:
                image_bytes = base64.b64decode(inline["data"])
                returned_mime = inline.get("mimeType", "image/png")
                break
            except Exception:
                continue

    if not image_bytes:
        raise RuntimeError(
            "Gemini did not return image bytes. Try a more specific prompt."
        )

    # Save output (keeps your original filename)
    output_path.write_bytes(image_bytes)
    print(f"Saved generated image to: {output_path.resolve()} (mime: {returned_mime})")


if __name__ == "__main__":
    main()
