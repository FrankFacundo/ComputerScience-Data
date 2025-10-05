#!/usr/bin/env python3
"""
Minimal Google Translate client (unofficial) using the public 'gtx' endpoint.

Notes:
- This uses an undocumented endpoint and may change or rate-limit you.
- No API key required. For production, use an official API.
"""

from urllib.parse import quote
import requests

class GoogleTranslate:
    def __init__(self, host="translate.googleapis.com", https=True):
        self.scheme = "https" if https else "http"
        self.host = host
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        })

    def _request_url(self, text, sl="auto", tl="en", hl=None, no_autocorrect=False):
        """
        Build the translate_a/single URL similar to the AWK code.
        """
        hl = hl or tl  # UI language for hints; mirror AWK behavior
        qc = "qc" if no_autocorrect else "qca"
        base = (f"{self.scheme}://{self.host}/translate_a/single?client=gtx"
                f"&ie=UTF-8&oe=UTF-8"
                f"&dt=bd&dt=ex&dt=ld&dt=md&dt=rw&dt=rm&dt=ss&dt=t&dt=at&dt=gt"
                f"&dt={qc}"
                f"&sl={quote(sl)}&tl={quote(tl)}&hl={quote(hl)}"
                f"&q={quote(text)}")
        return base

    def translate(self, text, sl="auto", tl="en", hl=None, no_autocorrect=False):
        """
        Return a dict with:
          - translation: str
          - original: str
          - src_lang: detected source language (ISO code)
          - alternatives: list of strings (if present)
          - raw: the raw JSON payload
        """
        url = self._request_url(text, sl=sl, tl=tl, hl=hl, no_autocorrect=no_autocorrect)
        r = self.session.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()

        # data[0] is a list of segments: [ [translated, original, ...], ... ]
        segments = data[0] or []
        translated_chunks = [seg[0] for seg in segments if seg and seg[0] is not None]
        original_chunks   = [seg[1] for seg in segments if seg and len(seg) > 1 and seg[1] is not None]

        translation = "".join(translated_chunks)
        original = "".join(original_chunks)

        # data[2] is detected source language (e.g., "es")
        src_lang = None
        if len(data) > 2 and isinstance(data[2], str):
            src_lang = data[2]

        # Alternative translations (when available) live under data[5]
        alternatives = []
        if len(data) > 5 and isinstance(data[5], list):
            # data[5] structure varies; try to pull the strings safely
            for entry in data[5]:
                try:
                    if isinstance(entry, list) and len(entry) > 2 and isinstance(entry[2], list):
                        for alt in entry[2]:
                            if isinstance(alt, list) and alt and isinstance(alt[0], str):
                                alternatives.append(alt[0])
                except Exception:
                    pass

        return {
            "translation": translation,
            "original": original or text,
            "src_lang": src_lang,
            "alternatives": list(dict.fromkeys(alternatives)),  # unique, keep order
            "raw": data,
        }

    def tts_url(self, text, tl="en"):
        """
        Build a TTS URL (mp3) like the AWK googleTTSUrl.
        (You fetch it yourself; this just returns the URL.)
        """
        return (f"{self.scheme}://{self.host}/translate_tts?ie=UTF-8&client=gtx"
                f"&tl={quote(tl)}&q={quote(text)}")

    def web_translate_url(self, url_to_translate, sl="auto", tl="en", hl=None):
        """
        Web UI wrapper URL, equivalent to googleWebTranslateUrl.
        """
        hl = hl or tl
        return (f"https://translate.google.com/translate?"
                f"hl={quote(hl)}&sl={quote(sl)}&tl={quote(tl)}&u={quote(url_to_translate)}")


if __name__ == "__main__":
    gt = GoogleTranslate()

    # --- Example: translate a phrase ---
    phrase = "Hola, ¿cómo estás?"
    result = gt.translate(phrase, sl="auto", tl="en", hl="en")

    print("Original:", result["original"])
    print("Detected source language:", result["src_lang"])
    print("Translation:", result["translation"])
    if result["alternatives"]:
        print("Alternatives:", ", ".join(result["alternatives"][:5]))

    # Optional: get a TTS URL for the translation
    print("TTS (mp3) URL:", gt.tts_url(result["translation"], tl="en"))
