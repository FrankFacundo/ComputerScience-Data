from googletrans import Translator
from pydantic import BaseModel


class TranslatePayload(BaseModel):
    text: str
    src: str
    dst: str


def translate(payload: TranslatePayload):
    translator = Translator()
    result = translator.translate(text=payload.text,
                                  dest=payload.dst,
                                  src=payload.src)
    return result.text