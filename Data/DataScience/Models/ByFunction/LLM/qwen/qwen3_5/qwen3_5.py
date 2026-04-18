import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoTokenizer


DEFAULT_MODEL_PATH = "/Users/frankfacundo/Models/Qwen/Qwen3.5-27B"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_PROMPT = "Describe this image."
IMAGE_TOKEN = "<|image_pad|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text + image inference with a local Qwen3.5 Hugging Face checkpoint."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", default=DEFAULT_USER_PROMPT, help="User text prompt.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System instruction for the chat template.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Local path to the Qwen3.5 model directory.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Ask the chat template for a direct answer without the thinking trace.",
    )
    return parser.parse_args()


def load_model(model_path: str):
    model_kwargs = {"trust_remote_code": True}
    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model_kwargs["torch_dtype"] = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        model_kwargs["torch_dtype"] = torch.float32

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    except ValueError as exc:
        if "model type `qwen3_5`" in str(exc):
            raise RuntimeError(
                "The installed `transformers` build does not recognize `qwen3_5`. "
                "Upgrade to a version that supports Qwen3.5 before running this script."
            ) from exc
        raise

    model = model.eval()

    if device.type != "cuda":
        model = model.to(device)

    return model


def build_messages(image_path: Path, prompt: str, system_prompt: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def expand_image_placeholders(
    prompt_text: str, image_grid_thw: torch.Tensor, merge_size: int
) -> str:
    expanded_text = prompt_text
    merge_length = merge_size**2

    for grid in image_grid_thw.tolist():
        num_image_tokens = (grid[0] * grid[1] * grid[2]) // merge_length
        expanded_text = expanded_text.replace(IMAGE_TOKEN, IMAGE_TOKEN * num_image_tokens, 1)

    return expanded_text


def prepare_inputs(
    *,
    model,
    tokenizer,
    image_processor,
    messages: list[dict],
    image_path: Path,
    disable_thinking: bool,
):
    chat_template_kwargs = {"enable_thinking": False} if disable_thinking else None
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )

    image = Image.open(image_path).convert("RGB")
    image_inputs = image_processor(images=image, return_tensors="pt")
    prompt_text = expand_image_placeholders(
        prompt_text=prompt_text,
        image_grid_thw=image_inputs["image_grid_thw"],
        merge_size=image_processor.merge_size,
    )

    text_inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {**text_inputs, **image_inputs}

    prepared_inputs = {}
    for name, value in inputs.items():
        if torch.is_floating_point(value):
            prepared_inputs[name] = value.to(device=model.device, dtype=model.dtype)
        else:
            prepared_inputs[name] = value.to(model.device)

    return prepared_inputs


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    messages = build_messages(
        image_path=image_path,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
    )
    inputs = prepare_inputs(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        messages=messages,
        image_path=image_path,
        disable_thinking=args.disable_thinking,
    )

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(decoded[0].strip())


if __name__ == "__main__":
    main()
