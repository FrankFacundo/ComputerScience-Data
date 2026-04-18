"""Pure-python Qwen2-style BBPE tokenizer (also what Qwen3.5 uses).

Pipeline matches the HuggingFace tokenizers `Qwen2Tokenizer` backend:

    special-token split  ->  NFC normalize  ->  pretokenize (regex)
        ->  byte-level encode  ->  BPE merge  ->  vocab lookup

Decoding is the inverse: id -> byte-level string -> utf-8 bytes -> text.

External deps: `regex` (needed for `\\p{L}` Unicode categories in the
pretokenization regex).
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import regex as re


# -- GPT-2 bytes ↔ unicode mapping (same as transformers / tokenizers) ----


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """Map every byte to a printable, non-whitespace unicode codepoint."""
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


# -- BPE merging over byte-encoded "words" --------------------------------


class _BPE:
    """Plain BPE merger over string tokens."""

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


# -- public tokenizer -----------------------------------------------------


class Qwen2Tokenizer:
    """Pure-python Qwen2/Qwen3.5 BBPE tokenizer.

    Construction is cheap; all state is vocab/merges/special-token tables.
    Encoded pieces are compared byte-for-byte to transformers output.
    """

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
    ):
        self.add_prefix_space = add_prefix_space
        self.errors = errors

        self._bpe = _BPE(vocab, merges)
        self._byte_encoder = bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

        self._added: dict[str, int] = dict(added_tokens or {})
        self._added_decoder = {v: k for k, v in self._added.items()}

        pattern = pretokenize_regex or self._DEFAULT_PRETOKENIZE_REGEX
        self._pretok_re = re.compile(pattern)

        if self._added:
            escaped = sorted(
                (re.escape(t) for t in self._added.keys()),
                key=len,
                reverse=True,
            )
            self._special_re: re.Pattern | None = re.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_re = None

    # -- loaders --------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Qwen2Tokenizer":
        model_dir = Path(model_dir)
        with open(model_dir / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(model_dir / "merges.txt", encoding="utf-8") as f:
            merges_lines = f.read().split("\n")
        # drop header line "#version: 0.2" and empty trailing lines
        merges: list[tuple[str, str]] = []
        for line in merges_lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            merges.append((parts[0], parts[1]))

        cfg_path = model_dir / "tokenizer_config.json"
        added_tokens: dict[str, int] = {}
        pretokenize_regex: str | None = None
        errors = "replace"
        add_prefix_space = False
        chat_template: str | None = None
        special_info: dict[str, dict] = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            for id_str, info in cfg.get("added_tokens_decoder", {}).items():
                added_tokens[info["content"]] = int(id_str)
                special_info[info["content"]] = info
            pretokenize_regex = cfg.get("pretokenize_regex")
            errors = cfg.get("errors", "replace")
            add_prefix_space = bool(cfg.get("add_prefix_space", False))
            chat_template = cfg.get("chat_template")

        tok = cls(
            vocab,
            merges,
            added_tokens=added_tokens,
            pretokenize_regex=pretokenize_regex,
            errors=errors,
            add_prefix_space=add_prefix_space,
        )
        tok._special_info = special_info
        tok._chat_template = chat_template
        tok._config = cfg if cfg_path.exists() else {}
        return tok

    # -- public API -----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._bpe.encoder) + len(self._added)

    @property
    def eos_token(self) -> str | None:
        return self._config.get("eos_token")

    @property
    def eos_token_id(self) -> int | None:
        t = self.eos_token
        return self._added.get(t) if t else None

    @property
    def pad_token(self) -> str | None:
        return self._config.get("pad_token")

    @property
    def pad_token_id(self) -> int | None:
        t = self.pad_token
        return self._added.get(t) if t else None

    @property
    def chat_template(self) -> str | None:
        return getattr(self, "_chat_template", None)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode `text` to token ids. `add_special_tokens` is a no-op for Qwen2."""
        return list(self._encode_iter(text))

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = False) -> str:
        tokens: list[str] = []
        for i in ids:
            i = int(i)
            if i in self._added_decoder:
                if skip_special_tokens and self._is_special(i):
                    continue
                tokens.append(self._added_decoder[i])
            else:
                tokens.append(self._bpe.decoder[i])
        # group consecutive non-special tokens so we decode bytes contiguously
        out: list[str] = []
        buf: list[str] = []
        for t in tokens:
            if t in self._added:
                if buf:
                    out.append(self._decode_bytes(buf))
                    buf = []
                out.append(t)
            else:
                buf.append(t)
        if buf:
            out.append(self._decode_bytes(buf))
        text = "".join(out)
        if clean_up_tokenization_spaces:
            text = self._cleanup_spaces(text)
        return text

    def __call__(self, text: str, *, return_tensors: str | None = None) -> dict:
        ids = self.encode(text)
        mask = [1] * len(ids)
        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long),
            }
        return {"input_ids": [ids], "attention_mask": [mask]}

    # -- chat template --------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        chat_template_kwargs: dict | None = None,
        **extra_kwargs,
    ) -> str | list[int]:
        """Render the stored chat template (Jinja2) with `messages`.

        Any unknown keyword args (e.g. `enable_thinking`) are forwarded to the
        template, matching transformers' behavior. `chat_template_kwargs`, if
        provided, is merged in after direct kwargs.
        """
        if not self._chat_template:
            raise ValueError("No chat template configured on this tokenizer.")
        import jinja2

        env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )

        def _raise(msg: str):
            raise ValueError(msg)

        env.globals["raise_exception"] = _raise
        template = env.from_string(self._chat_template)

        kwargs = {
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "add_vision_id": False,
        }
        kwargs.update(extra_kwargs)
        if chat_template_kwargs:
            kwargs.update(chat_template_kwargs)
        rendered = template.render(**kwargs)

        if tokenize:
            return self.encode(rendered)
        return rendered

    # -- internals ------------------------------------------------------

    def _is_special(self, token_id: int) -> bool:
        tok = self._added_decoder.get(token_id)
        if tok is None:
            return False
        info = getattr(self, "_special_info", {}).get(tok, {})
        return bool(info.get("special", False))

    def _encode_iter(self, text: str) -> Iterable[int]:
        if self._special_re is None:
            yield from self._encode_chunk(text)
            return
        # split by special tokens, keeping them
        last = 0
        for m in self._special_re.finditer(text):
            if m.start() > last:
                yield from self._encode_chunk(text[last:m.start()])
            yield self._added[m.group(0)]
            last = m.end()
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
            # byte-level encode
            encoded = "".join(self._byte_encoder[b] for b in piece.encode("utf-8"))
            merged = self._bpe.bpe(encoded).split(" ")
            for sub in merged:
                yield self._bpe.encoder[sub]

    def _decode_bytes(self, tokens: list[str]) -> str:
        text = "".join(tokens)
        buf = bytearray(self._byte_decoder[c] for c in text)
        return buf.decode("utf-8", errors=self.errors)

    @staticmethod
    def _cleanup_spaces(s: str) -> str:
        for punct in (".", "?", "!", ",", "'", "n't", "'m", "'s", "'ve", "'re"):
            s = s.replace(f" {punct}", punct)
        return s
