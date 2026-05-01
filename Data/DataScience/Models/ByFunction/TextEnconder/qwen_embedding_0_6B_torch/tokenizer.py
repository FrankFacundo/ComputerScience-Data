"""Pure-python Qwen2/Qwen3 byte-level BPE tokenizer."""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import regex as re


class BatchEncoding(dict):
    """Small dict subclass with the transformers-style .to(device) helper."""

    def to(self, device):
        for key, value in self.items():
            if hasattr(value, "to"):
                self[key] = value.to(device)
        return self


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(a, b) for a, b in zip(word, word[1:])}


class _BPE:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self._cache: dict[str, str] = {}

    def bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            return token
        while True:
            pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if pair not in self.bpe_ranks:
                break
            first, second = pair
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self._cache[token] = out
        return out


class Qwen2Tokenizer:
    """Tokenizer compatible with Qwen2/Qwen3 vocab.json + merges.txt files."""

    _DEFAULT_PRETOKENIZE_REGEX = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|"
        r"\p{N}|"
        r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        added_tokens: dict[str, int] | None = None,
        pretokenize_regex: str | None = None,
        errors: str = "replace",
        add_prefix_space: bool = False,
        padding_side: str = "right",
        truncation_side: str = "right",
    ):
        self.add_prefix_space = add_prefix_space
        self.errors = errors
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self._bpe = _BPE(vocab, merges)
        self._byte_encoder = bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        self._added = dict(added_tokens or {})
        self._added_decoder = {v: k for k, v in self._added.items()}
        self._special_info: dict[str, dict] = {}
        self._config: dict = {}
        self._chat_template: str | None = None
        self._post_processor_ids: list[int] = []

        self._pretok_re = re.compile(pretokenize_regex or self._DEFAULT_PRETOKENIZE_REGEX)
        if self._added:
            escaped = sorted((re.escape(t) for t in self._added), key=len, reverse=True)
            self._special_re: re.Pattern | None = re.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_re = None

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        padding_side: str | None = None,
    ) -> "Qwen2Tokenizer":
        model_dir = Path(model_dir)
        with open(model_dir / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(model_dir / "merges.txt", encoding="utf-8") as f:
            merges_lines = f.read().splitlines()
        merges = []
        for line in merges_lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

        cfg: dict = {}
        added_tokens: dict[str, int] = {}
        special_info: dict[str, dict] = {}
        pretokenize_regex = None
        errors = "replace"
        add_prefix_space = False
        truncation_side = "right"
        cfg_path = model_dir / "tokenizer_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            for id_str, info in cfg.get("added_tokens_decoder", {}).items():
                added_tokens[info["content"]] = int(id_str)
                special_info[info["content"]] = info
            pretokenize_regex = cfg.get("pretokenize_regex")
            errors = cfg.get("errors", "replace")
            add_prefix_space = bool(cfg.get("add_prefix_space", False))
            truncation_side = cfg.get("truncation_side") or "right"
            if padding_side is None:
                padding_side = cfg.get("padding_side")

        post_processor_ids = cls._load_post_processor_ids(model_dir, added_tokens)
        tok = cls(
            vocab,
            merges,
            added_tokens=added_tokens,
            pretokenize_regex=pretokenize_regex,
            errors=errors,
            add_prefix_space=add_prefix_space,
            padding_side=padding_side or "right",
            truncation_side=truncation_side,
        )
        tok._special_info = special_info
        tok._config = cfg
        tok._chat_template = cfg.get("chat_template")
        tok._post_processor_ids = post_processor_ids
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self._bpe.encoder) + len(self._added)

    @property
    def pad_token(self) -> str | None:
        return self._token_content(self._config.get("pad_token"))

    @property
    def pad_token_id(self) -> int | None:
        token = self.pad_token
        return self._added.get(token) if token is not None else None

    @property
    def eos_token(self) -> str | None:
        return self._token_content(self._config.get("eos_token"))

    @property
    def eos_token_id(self) -> int | None:
        token = self.eos_token
        return self._added.get(token) if token is not None else None

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = list(self._encode_iter(text))
        if add_special_tokens:
            ids.extend(self._post_processor_ids)
        return ids

    def decode(
        self,
        ids: Iterable[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        tokens: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self._added_decoder:
                if skip_special_tokens and self._is_special(token_id):
                    continue
                tokens.append(self._added_decoder[token_id])
            else:
                tokens.append(self._bpe.decoder[token_id])

        out: list[str] = []
        buf: list[str] = []
        for token in tokens:
            if token in self._added:
                if buf:
                    out.append(self._decode_bytes(buf))
                    buf = []
                out.append(token)
            else:
                buf.append(token)
        if buf:
            out.append(self._decode_bytes(buf))
        text = "".join(out)
        if clean_up_tokenization_spaces:
            text = self._cleanup_spaces(text)
        return text

    def __call__(
        self,
        text: str | list[str],
        *,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> BatchEncoding:
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [self.encode(t, add_special_tokens=False) for t in texts]
        if truncation and max_length is not None:
            special_len = len(self._post_processor_ids) if add_special_tokens else 0
            keep = max(max_length - special_len, 0)
            if self.truncation_side == "left":
                encoded = [ids[-keep:] if keep else [] for ids in encoded]
            else:
                encoded = [ids[:keep] for ids in encoded]
        if add_special_tokens:
            encoded = [ids + list(self._post_processor_ids) for ids in encoded]

        target_length = None
        if padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required when padding='max_length'")
            target_length = max_length
        elif padding:
            target_length = max(len(ids) for ids in encoded) if encoded else 0

        attention_masks: list[list[int]] = []
        if target_length is not None:
            pad_id = self.pad_token_id
            if pad_id is None:
                raise ValueError("Tokenizer has no pad_token_id.")
            padded = []
            for ids in encoded:
                pad_len = max(0, target_length - len(ids))
                if self.padding_side == "left":
                    padded.append([pad_id] * pad_len + ids)
                    attention_masks.append([0] * pad_len + [1] * len(ids))
                else:
                    padded.append(ids + [pad_id] * pad_len)
                    attention_masks.append([1] * len(ids) + [0] * pad_len)
            encoded = padded
        else:
            attention_masks = [[1] * len(ids) for ids in encoded]

        out = BatchEncoding({"input_ids": encoded, "attention_mask": attention_masks})
        if return_tensors == "pt":
            import torch

            out["input_ids"] = torch.tensor(encoded, dtype=torch.long)
            out["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
        elif return_tensors is not None:
            raise ValueError("Only return_tensors='pt' is supported.")
        return out

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        chat_template_kwargs: dict | None = None,
        **extra_kwargs,
    ) -> str | list[int]:
        if not self._chat_template:
            raise ValueError("No chat template configured on this tokenizer.")
        import jinja2

        env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        template = env.from_string(self._chat_template)
        kwargs = {"messages": messages, "add_generation_prompt": add_generation_prompt}
        kwargs.update(extra_kwargs)
        if chat_template_kwargs:
            kwargs.update(chat_template_kwargs)
        rendered = template.render(**kwargs)
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered

    def _encode_iter(self, text: str) -> Iterable[int]:
        if self._special_re is None:
            yield from self._encode_chunk(text)
            return
        last = 0
        for match in self._special_re.finditer(text):
            if match.start() > last:
                yield from self._encode_chunk(text[last : match.start()])
            yield self._added[match.group(0)]
            last = match.end()
        if last < len(text):
            yield from self._encode_chunk(text[last:])

    def _encode_chunk(self, text: str) -> Iterable[int]:
        if not text:
            return
        text = unicodedata.normalize("NFC", text)
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text
        for piece in self._pretok_re.findall(text):
            if not piece:
                continue
            encoded = "".join(self._byte_encoder[b] for b in piece.encode("utf-8"))
            for sub in self._bpe.bpe(encoded).split(" "):
                yield self._bpe.encoder[sub]

    def _decode_bytes(self, tokens: list[str]) -> str:
        text = "".join(tokens)
        buf = bytearray(self._byte_decoder[c] for c in text)
        return buf.decode("utf-8", errors=self.errors)

    def _is_special(self, token_id: int) -> bool:
        token = self._added_decoder.get(token_id)
        if token is None:
            return False
        return bool(self._special_info.get(token, {}).get("special", False))

    @staticmethod
    def _token_content(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("content")
        return str(value)

    @staticmethod
    def _load_post_processor_ids(model_dir: Path, added_tokens: dict[str, int]) -> list[int]:
        tokenizer_json = model_dir / "tokenizer.json"
        if not tokenizer_json.exists():
            return []
        with open(tokenizer_json, encoding="utf-8") as f:
            data = json.load(f)

        processor = data.get("post_processor")
        if not processor:
            return []
        processors = processor.get("processors", [processor]) if isinstance(processor, dict) else []
        for item in processors:
            if not isinstance(item, dict) or item.get("type") != "TemplateProcessing":
                continue
            ids: list[int] = []
            special_tokens = item.get("special_tokens", {})
            for entry in item.get("single", []):
                special = entry.get("SpecialToken") if isinstance(entry, dict) else None
                if not special:
                    continue
                token = special.get("id")
                info = special_tokens.get(token, {})
                if "ids" in info:
                    ids.extend(int(i) for i in info["ids"])
                elif token in added_tokens:
                    ids.append(added_tokens[token])
            return ids
        return []

    @staticmethod
    def _cleanup_spaces(text: str) -> str:
        for punct in (".", "?", "!", ",", "'", "n't", "'m", "'s", "'ve", "'re"):
            text = text.replace(f" {punct}", punct)
        return text
