#!/usr/bin/env python3
"""Generate static HTML API docs from Python source without importing modules.

This is intentionally AST-based rather than pydoc-based so it can document
modules that depend on optional packages or heavyweight runtimes.
"""

from __future__ import annotations

import argparse
import ast
import html
import inspect
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
  import markdown as markdown_lib
except ImportError:
  markdown_lib = None


PAGE_CSS = """
body {
  margin: 0;
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f4ee;
  color: #1f2933;
}
a {
  color: #0f4c5c;
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
.page {
  max-width: 1100px;
  margin: 0 auto;
  padding: 32px 24px 64px;
}
.eyebrow {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 999px;
  background: #d7e7ea;
  color: #0f4c5c;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
h1, h2, h3 {
  margin: 0;
  line-height: 1.2;
}
h1 {
  margin-top: 12px;
  font-size: 36px;
}
h2 {
  margin-top: 36px;
  font-size: 24px;
}
h3 {
  margin-top: 24px;
  font-size: 18px;
}
.subtitle {
  margin-top: 8px;
  color: #52606d;
}
.toc {
  margin-top: 28px;
  padding: 16px 18px;
  background: #fffdf7;
  border: 1px solid #e5dfd0;
  border-radius: 14px;
}
.toc ul {
  margin: 12px 0 0;
  padding-left: 20px;
}
.section {
  margin-top: 28px;
  padding: 22px 22px 18px;
  background: #fff;
  border: 1px solid #ebe5d6;
  border-radius: 18px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.meta {
  margin-top: 8px;
  color: #52606d;
  font-size: 14px;
}
.docstring {
  margin-top: 14px;
  padding: 14px 16px;
  background: #f8faf9;
  border-left: 4px solid #d7e7ea;
  border-radius: 10px;
  overflow-x: auto;
  line-height: 1.6;
}
.docstring > :first-child {
  margin-top: 0;
}
.docstring > :last-child {
  margin-bottom: 0;
}
.docstring p,
.docstring ul,
.docstring ol,
.docstring blockquote {
  margin: 0 0 14px;
}
.docstring ul,
.docstring ol {
  padding-left: 24px;
}
.docstring li {
  margin: 6px 0;
}
.docstring pre {
  margin: 0 0 14px;
  padding: 12px 14px;
  white-space: pre-wrap;
  word-break: break-word;
  background: #eef4f6;
  border-radius: 8px;
  font: 14px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
}
.docstring code {
  padding: 0.1em 0.35em;
  background: #eef4f6;
  border-radius: 4px;
}
.docstring pre code {
  padding: 0;
  background: transparent;
}
.docstring h1,
.docstring h2,
.docstring h3,
.docstring h4 {
  margin: 22px 0 10px;
  font-size: 1.1em;
}
.docstring hr {
  border: 0;
  border-top: 1px solid #d9e2e8;
  margin: 18px 0;
}
.signature {
  margin-top: 12px;
  padding: 12px 14px;
  background: #13293d;
  color: #f4f7fb;
  border-radius: 10px;
  overflow-x: auto;
  font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
}
.member-list {
  margin-top: 12px;
}
.member-list li {
  margin: 8px 0;
}
.small {
  font-size: 14px;
  color: #52606d;
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
"""


@dataclass
class FunctionDoc:
  name: str
  signature: str
  decorators: list[str] = field(default_factory=list)
  docstring: str = ""
  lineno: int = 0


@dataclass
class ClassDoc:
  name: str
  bases: list[str]
  docstring: str = ""
  lineno: int = 0
  methods: list[FunctionDoc] = field(default_factory=list)


@dataclass
class ConstantDoc:
  name: str
  value: str
  lineno: int = 0


@dataclass
class ModuleDoc:
  source_path: Path
  rel_source_path: Path
  module_name: str
  docstring: str
  constants: list[ConstantDoc]
  functions: list[FunctionDoc]
  classes: list[ClassDoc]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Generate static HTML docs from Python source files.")
  parser.add_argument("paths", nargs="+", help="Python files or package directories to document.")
  parser.add_argument(
      "--output-dir",
      required=True,
      help="Directory that will receive the generated HTML files.",
  )
  return parser.parse_args()


def discover_python_files(paths: Iterable[str]) -> list[Path]:
  files: list[Path] = []
  for raw_path in paths:
    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
      if path.suffix == ".py":
        files.append(path)
      continue
    if path.is_dir():
      for file_path in sorted(path.rglob("*.py")):
        if "__pycache__" in file_path.parts:
          continue
        files.append(file_path.resolve())
  deduped: dict[Path, None] = {}
  for file_path in files:
    deduped[file_path] = None
  return sorted(deduped)


def build_module_name(source_path: Path, rel_source_path: Path) -> str:
  parts = list(rel_source_path.parts)
  if parts[-1] == "__init__.py":
    parts = parts[:-1]
  else:
    parts[-1] = rel_source_path.stem
  return ".".join(parts)


def is_public_name(name: str) -> bool:
  return not name.startswith("_") or name in {"__init__"}


def expr_to_text(node: ast.AST | None) -> str:
  if node is None:
    return ""
  try:
    return ast.unparse(node)
  except Exception:
    return "<unparseable>"


def docstring_expr(node: ast.AST) -> ast.Expr | None:
  body = getattr(node, "body", None)
  if not body:
    return None
  first = body[0]
  if not isinstance(first, ast.Expr):
    return None
  value = first.value
  if isinstance(value, ast.Constant) and isinstance(value.value, str):
    return first
  return None


def raw_docstring(node: ast.AST, source: str) -> str:
  expr = docstring_expr(node)
  if expr is None:
    return ""
  literal = ast.get_source_segment(source, expr.value)
  if literal is None:
    try:
      return ast.get_docstring(node) or ""
    except TypeError:
      return ""
  index = 0
  while index < len(literal) and literal[index] in "rRuUbBfF":
    index += 1
  for quote in ('"""', "'''", '"', "'"):
    if literal[index:index + len(quote)] == quote and literal.endswith(quote):
      inner = literal[index + len(quote):-len(quote)]
      return inspect.cleandoc(inner)
  try:
    return ast.get_docstring(node) or ""
  except TypeError:
    return ""


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
  args = []
  positional = list(node.args.posonlyargs) + list(node.args.args)
  positional_defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
  for index, arg in enumerate(node.args.posonlyargs):
    default = positional_defaults[index]
    args.append(format_arg(arg, default))
  if node.args.posonlyargs:
    args.append("/")
  for offset, arg in enumerate(node.args.args, start=len(node.args.posonlyargs)):
    default = positional_defaults[offset]
    args.append(format_arg(arg, default))
  if node.args.vararg is not None:
    args.append("*" + format_arg(node.args.vararg, None, include_annotation=True, include_name_only=True))
  elif node.args.kwonlyargs:
    args.append("*")
  for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
    args.append(format_arg(arg, default))
  if node.args.kwarg is not None:
    args.append("**" + format_arg(node.args.kwarg, None, include_annotation=True, include_name_only=True))
  returns = f" -> {expr_to_text(node.returns)}" if node.returns is not None else ""
  prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
  return f"{prefix}{node.name}({', '.join(args)}){returns}"


def format_arg(
    arg: ast.arg,
    default: ast.AST | None,
    *,
    include_annotation: bool = True,
    include_name_only: bool = False,
) -> str:
  text = arg.arg
  if include_annotation and arg.annotation is not None:
    text += f": {expr_to_text(arg.annotation)}"
  if include_name_only:
    return text
  if default is not None:
    text += f" = {expr_to_text(default)}"
  return text


def format_class_signature(node: ast.ClassDef) -> str:
  bases = [expr_to_text(base) for base in node.bases]
  if bases:
    return f"class {node.name}({', '.join(bases)})"
  return f"class {node.name}"


def extract_constants(nodes: list[ast.stmt]) -> list[ConstantDoc]:
  constants: list[ConstantDoc] = []
  for node in nodes:
    if isinstance(node, ast.Assign):
      if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        continue
      name = node.targets[0].id
      if not name.isupper():
        continue
      constants.append(
          ConstantDoc(name=name, value=expr_to_text(node.value), lineno=getattr(node, "lineno", 0))
      )
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      name = node.target.id
      if not name.isupper():
        continue
      constants.append(
          ConstantDoc(name=name, value=expr_to_text(node.value), lineno=getattr(node, "lineno", 0))
      )
  return constants


def extract_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionDoc:
  return FunctionDoc(
      name=node.name,
      signature=format_signature(node),
      decorators=[expr_to_text(decorator) for decorator in node.decorator_list],
      docstring="",
      lineno=getattr(node, "lineno", 0),
  )


def extract_class(node: ast.ClassDef, source: str) -> ClassDoc:
  methods = []
  for child in node.body:
    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public_name(child.name):
      method_doc = extract_function(child)
      method_doc.docstring = raw_docstring(child, source)
      methods.append(method_doc)
  return ClassDoc(
      name=node.name,
      bases=[expr_to_text(base) for base in node.bases],
      docstring=raw_docstring(node, source),
      lineno=getattr(node, "lineno", 0),
      methods=methods,
  )


def parse_module(source_path: Path, rel_source_path: Path) -> ModuleDoc:
  source = source_path.read_text(encoding="utf-8")
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    tree = ast.parse(source, filename=str(source_path))
  constants = extract_constants(tree.body)
  functions = []
  classes = []
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public_name(node.name):
      function_doc = extract_function(node)
      function_doc.docstring = raw_docstring(node, source)
      functions.append(function_doc)
    elif isinstance(node, ast.ClassDef) and is_public_name(node.name):
      classes.append(extract_class(node, source))
  return ModuleDoc(
      source_path=source_path,
      rel_source_path=rel_source_path,
      module_name=build_module_name(source_path, rel_source_path),
      docstring=raw_docstring(tree, source),
      constants=constants,
      functions=functions,
      classes=classes,
  )


def h(text: str) -> str:
  return html.escape(text)


def normalize_docstring_markup(text: str) -> str:
  text = text.replace("\r\n", "\n")
  text = re.sub(r":(?:func|meth|class|mod|attr|obj|data):`([^`]+)`", r"`\1`", text)
  text = re.sub(r":math:`([^`]+)`", r"`\1`", text)
  text = re.sub(r"(?m)^([A-Za-z][^\n]{0,120})\n([=\-~`^:#]{3,})\s*$", _heading_replacement, text)
  text = re.sub(r"(?m)::[ \t]*$", ":", text)
  return text


def _heading_replacement(match: re.Match[str]) -> str:
  title = match.group(1).strip()
  underline = match.group(2).strip()
  level_map = {"=": "#", "-": "##", "~": "###", "`": "####", "^": "####", ":": "####", "#": "####"}
  prefix = level_map.get(underline[0], "##")
  return f"{prefix} {title}"


def render_markdown(text: str) -> str:
  normalized = normalize_docstring_markup(text)
  if markdown_lib is not None:
    return markdown_lib.markdown(
        normalized,
        extensions=[
            "extra",
            "sane_lists",
            "fenced_code",
        ],
        output_format="html5",
    )
  return render_markdown_builtin(normalized)


_BLOCK_START_RE = re.compile(
    r"^(?:"
    r"#{1,6}\s|"
    r"```|"
    r"~~~|"
    r"\s*[-*+]\s+|"
    r"\s*\d+\.\s+|"
    r"\s*([-*_])\s*(?:\1\s*){2,}$"
    r")"
)


def _is_block_start(line: str) -> bool:
  return bool(_BLOCK_START_RE.match(line))


def render_markdown_builtin(text: str) -> str:
  lines = text.split("\n")
  i = 0
  out: list[str] = []
  n = len(lines)
  while i < n:
    if not lines[i].strip():
      i += 1
      continue

    line = lines[i]

    fence_match = re.match(r"^(\s*)(```|~~~)\s*(\S*)\s*$", line)
    if fence_match:
      fence = fence_match.group(2)
      lang = fence_match.group(3)
      i += 1
      code_lines: list[str] = []
      while i < n and not re.match(rf"^\s*{re.escape(fence)}\s*$", lines[i]):
        code_lines.append(lines[i])
        i += 1
      if i < n:
        i += 1
      out.append(_render_code_block("\n".join(code_lines), lang))
      continue

    heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
    if heading:
      level = len(heading.group(1))
      out.append(f"<h{level}>{render_inline(heading.group(2))}</h{level}>")
      i += 1
      continue

    if re.match(r"^\s*([-*_])\s*(?:\1\s*){2,}$", line):
      out.append("<hr>")
      i += 1
      continue

    if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
      list_html, i = _parse_list(lines, i)
      out.append(list_html)
      continue

    if re.match(r"^(?: {4}|\t)", line):
      code_lines = []
      while i < n:
        peek = lines[i]
        if re.match(r"^(?: {4}|\t)", peek):
          code_lines.append(re.sub(r"^(?: {4}|\t)", "", peek))
          i += 1
          continue
        if not peek.strip():
          j = i + 1
          while j < n and not lines[j].strip():
            j += 1
          if j < n and re.match(r"^(?: {4}|\t)", lines[j]):
            code_lines.append("")
            i += 1
            continue
        break
      while code_lines and not code_lines[-1].strip():
        code_lines.pop()
      out.append(_render_code_block("\n".join(code_lines), ""))
      continue

    para_lines = [line]
    i += 1
    while i < n:
      peek = lines[i]
      if not peek.strip():
        break
      if _is_block_start(peek):
        break
      para_lines.append(peek)
      i += 1
    joined = " ".join(item.strip() for item in para_lines)
    out.append(f"<p>{render_inline(joined)}</p>")

  return "\n".join(out)


def _render_code_block(code: str, lang: str) -> str:
  class_attr = f' class="language-{h(lang)}"' if lang else ""
  return f"<pre><code{class_attr}>{h(code)}</code></pre>"


def _parse_list(lines: list[str], start: int) -> tuple[str, int]:
  n = len(lines)
  first = lines[start]
  head = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", first)
  if head is None:
    return "", start + 1
  base_indent = len(head.group(1))
  ordered = head.group(2)[0].isdigit()
  marker_len = len(head.group(2)) + 1

  items: list[list[str]] = []
  current: list[str] = []
  i = start

  while i < n:
    line = lines[i]
    if not line.strip():
      j = i + 1
      while j < n and not lines[j].strip():
        j += 1
      if j >= n:
        i = j
        break
      next_line = lines[j]
      next_indent = len(next_line) - len(next_line.lstrip())
      continues_list = (
          next_indent == base_indent
          and re.match(r"^\s*(?:[-*+]|\d+\.)\s+", next_line)
      ) or next_indent >= base_indent + marker_len
      if not continues_list:
        i = j
        break
      current.append("")
      i = j
      continue

    indent = len(line) - len(line.lstrip())
    marker_match = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)

    if marker_match and indent == base_indent:
      if current:
        items.append(current)
        current = []
      current.append(marker_match.group(3))
      i += 1
      continue

    if indent >= base_indent + marker_len:
      current.append(line[base_indent + marker_len:])
      i += 1
      continue

    if current and indent >= base_indent and not marker_match:
      current.append(line.strip())
      i += 1
      continue

    break

  if current:
    items.append(current)

  tag = "ol" if ordered else "ul"
  rendered_items = []
  for item_lines in items:
    while item_lines and not item_lines[-1].strip():
      item_lines.pop()
    has_block = any(not ln.strip() for ln in item_lines) or any(
        _is_block_start(ln) for ln in item_lines
    )
    if has_block:
      inner = render_markdown_builtin("\n".join(item_lines))
      rendered_items.append(f"<li>{inner}</li>")
    else:
      flat = " ".join(ln.strip() for ln in item_lines if ln.strip())
      rendered_items.append(f"<li>{render_inline(flat)}</li>")

  return f"<{tag}>{''.join(rendered_items)}</{tag}>", i


def render_inline(text: str) -> str:
  placeholders: list[str] = []

  def stash_code(match: re.Match[str]) -> str:
    placeholders.append(f"<code>{h(match.group(1))}</code>")
    return f"\x00CODE{len(placeholders) - 1}\x00"

  text = re.sub(r"``([^`]+?)``", stash_code, text)
  text = re.sub(r"`([^`]+?)`", stash_code, text)
  text = h(text)
  text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
  text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
  text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", text)
  text = re.sub(
      r"(?:(?<=^)|(?<=[^\w_]))_(?!\s)([^_\n]+?)(?<!\s)_(?=[^\w_]|$)",
      r"<em>\1</em>",
      text,
  )
  text = re.sub(
      r"\[([^\]]+)\]\(([^)\s]+)\)",
      lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
      text,
  )
  for index, replacement in enumerate(placeholders):
    text = text.replace(f"\x00CODE{index}\x00", replacement)
  return text


def doc_html(text: str) -> str:
  if not text:
    return '<p class="small">No docstring.</p>'
  return f'<div class="docstring">{render_markdown(text)}</div>'


def decorators_html(decorators: list[str]) -> str:
  if not decorators:
    return ""
  items = "".join(f"<li><code>@{h(decorator)}</code></li>" for decorator in decorators)
  return f'<div class="meta">Decorators</div><ul class="member-list">{items}</ul>'


def render_function(function_doc: FunctionDoc, heading_level: str = "h3") -> str:
  return f"""
  <section class="section" id="fn-{h(function_doc.name)}">
    <{heading_level}>{h(function_doc.name)}</{heading_level}>
    <div class="meta">Line {function_doc.lineno}</div>
    <div class="signature">{h(function_doc.signature)}</div>
    {decorators_html(function_doc.decorators)}
    {doc_html(function_doc.docstring)}
  </section>
  """


def render_class(class_doc: ClassDoc) -> str:
  bases = ", ".join(class_doc.bases) if class_doc.bases else "object"
  methods_html = ""
  if class_doc.methods:
    methods = "".join(render_function(method_doc, heading_level="h4") for method_doc in class_doc.methods)
    methods_html = f"<h3>Methods</h3>{methods}"
  return f"""
  <section class="section" id="class-{h(class_doc.name)}">
    <h2>{h(class_doc.name)}</h2>
    <div class="meta">Line {class_doc.lineno} · Bases: <code>{h(bases)}</code></div>
    <div class="signature">{h(format_class_signature_from_doc(class_doc))}</div>
    {doc_html(class_doc.docstring)}
    {methods_html}
  </section>
  """


def format_class_signature_from_doc(class_doc: ClassDoc) -> str:
  if class_doc.bases:
    return f"class {class_doc.name}({', '.join(class_doc.bases)})"
  return f"class {class_doc.name}"


def render_constants(constants: list[ConstantDoc]) -> str:
  if not constants:
    return ""
  items = "".join(
      f"<li><code>{h(constant.name)}</code> = <code>{h(constant.value)}</code> <span class=\"small\">(line {constant.lineno})</span></li>"
      for constant in constants
  )
  return f"""
  <section class="section" id="constants">
    <h2>Constants</h2>
    <ul class="member-list">{items}</ul>
  </section>
  """


def build_toc(module_doc: ModuleDoc) -> str:
  items = []
  if module_doc.constants:
    items.append('<li><a href="#constants">Constants</a></li>')
  if module_doc.classes:
    class_links = "".join(
        f'<li><a href="#class-{h(class_doc.name)}">{h(class_doc.name)}</a></li>' for class_doc in module_doc.classes
    )
    items.append(f"<li>Classes<ul>{class_links}</ul></li>")
  if module_doc.functions:
    function_links = "".join(
        f'<li><a href="#fn-{h(function_doc.name)}">{h(function_doc.name)}</a></li>'
        for function_doc in module_doc.functions
    )
    items.append(f"<li>Functions<ul>{function_links}</ul></li>")
  if not items:
    return ""
  return f"""
  <nav class="toc">
    <strong>Contents</strong>
    <ul>{''.join(items)}</ul>
  </nav>
  """


def render_module_page(module_doc: ModuleDoc, page_title: str) -> str:
  class_sections = "".join(render_class(class_doc) for class_doc in module_doc.classes)
  function_sections = "".join(render_function(function_doc) for function_doc in module_doc.functions)
  return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{h(page_title)}</title>
  <style>{PAGE_CSS}</style>
</head>
<body>
  <main class="page">
    <span class="eyebrow">Python Module</span>
    <h1>{h(module_doc.module_name)}</h1>
    <p class="subtitle">Source: <code>{h(str(module_doc.rel_source_path))}</code></p>
    {build_toc(module_doc)}
    <section class="section">
      <h2>Module Docstring</h2>
      {doc_html(module_doc.docstring)}
    </section>
    {render_constants(module_doc.constants)}
    {class_sections}
    {function_sections}
  </main>
</body>
</html>
"""


def render_index(modules: list[ModuleDoc], output_dir: Path) -> str:
  items = []
  for module_doc in modules:
    output_path = module_output_path(output_dir, module_doc.rel_source_path)
    href = output_path.relative_to(output_dir).as_posix()
    counts = []
    if module_doc.classes:
      counts.append(f"{len(module_doc.classes)} classes")
    if module_doc.functions:
      counts.append(f"{len(module_doc.functions)} functions")
    if module_doc.constants:
      counts.append(f"{len(module_doc.constants)} constants")
    summary = " · ".join(counts) if counts else "No public API symbols detected"
    items.append(
        f"""
        <section class="section">
          <h2><a href="{h(href)}">{h(module_doc.module_name)}</a></h2>
          <div class="meta"><code>{h(str(module_doc.rel_source_path))}</code></div>
          <div class="meta">{h(summary)}</div>
          {doc_html(module_doc.docstring)}
        </section>
        """
    )
  return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Python API Docs</title>
  <style>{PAGE_CSS}</style>
</head>
<body>
  <main class="page">
    <span class="eyebrow">Documentation</span>
    <h1>Python API Docs</h1>
    <p class="subtitle">Generated from source without importing runtime dependencies.</p>
    {''.join(items)}
  </main>
</body>
</html>
"""


def module_output_path(output_dir: Path, rel_source_path: Path) -> Path:
  if rel_source_path.name == "__init__.py":
    return output_dir.joinpath(*rel_source_path.parts[:-1], "index.html")
  return output_dir.joinpath(*rel_source_path.parts[:-1], f"{rel_source_path.stem}.html")


def write_module_page(output_dir: Path, module_doc: ModuleDoc) -> Path:
  output_path = module_output_path(output_dir, module_doc.rel_source_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
      render_module_page(module_doc, page_title=f"{module_doc.module_name} API docs"),
      encoding="utf-8",
  )
  return output_path


def main() -> None:
  args = parse_args()
  files = discover_python_files(args.paths)
  if not files:
    raise SystemExit("No Python files found.")

  common_root = Path(os.path.commonpath([str(file_path) for file_path in files]))
  if common_root.is_file():
    common_root = common_root.parent

  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  modules = []
  for file_path in files:
    rel_source_path = file_path.relative_to(common_root)
    modules.append(parse_module(file_path, rel_source_path))
  modules.sort(key=lambda module_doc: module_doc.module_name)

  for module_doc in modules:
    write_module_page(output_dir, module_doc)
  index_path = output_dir / "index.html"
  index_path.write_text(render_index(modules, output_dir), encoding="utf-8")

  print(f"Generated {len(modules)} module pages")
  print(f"Index: {index_path}")


if __name__ == "__main__":
  main()
