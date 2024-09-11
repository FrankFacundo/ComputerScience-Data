from mistral_common.protocol.instruct.messages import (
    ImageChunk,
    TextChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from PIL import Image

tokenizer = MistralTokenizer.from_model("pixtral")

image = Image.open("cartes.png")


# tokenize images and text
tokenized = tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="Transcribe this image into markdown"),
                    ImageChunk(image=image),
                ]
            )
        ],
        model="pixtral",
    )
)
tokens, text, images = tokenized.tokens, tokenized.text, tokenized.images

# Count the number of tokens
print("# tokens", len(tokens))
print("# images", len(images))
