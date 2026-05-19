## Use of Qwen3.5 with tranformers and will full torch.

```bash
python Data/DataScience/Models/ByFunction/LLM/qwen/qwen3_5.py \
  --image /path/to/image.png \
  --prompt "Describe this image in detail."
```

```bash
python qwen3_5.py \
  --image image.png \
  --prompt "Describe this image in detail."
```

```bash
python qwen3_5_torch.py \
    --image image.png \
    --prompt "Describe this image in detail."
```

## SAE Spanish steering

`sae.py` implements the Qwen-Scope feature-steering recipe for Spanish output:

```bash
python sae.py \
  --prompt "Tell me about recent advances in LLMs." \
  --layer 32 \
  --steering-strength 20 \
  --max-new-tokens 200
```

If the Spanish feature id for a layer is already known, skip contrastive
feature discovery:

```bash
python sae.py \
  --prompt "Tell me about recent advances in LLMs." \
  --layer 32 \
  --spanish-feature-id 62752 \
  --steering-strength 20 \
  --max-new-tokens 200
```

`62752` is the discovered Spanish feature for `layer32.sae.pt`; feature ids are
layer-local.

Example behavior with an English prompt and Spanish steering enabled:

```text
$ python sae.py \
  --prompt "Tell me about recent advances in LLMs." \
  --layer 32 \
  --spanish-feature-id 62752 \
  --steering-strength 20 \
  --max-new-tokens 200

[feat] using supplied feature id(s): [62752]
[gen]  steering strength=20.0 prompt_len=58 max_new_tokens=200

=== Spanish-steered completion ===
Thinking Process:

1.  **Analyze the Request:**
    *   **Usuario:** Pide informacion sobre los avances recientes en LLMs
        (Large Language Models).
    *   **Instruccion del Asistente:** Debo responder siempre en espanol,
        incluso si el usuario escribe en otro idioma (en este caso, ingles).
        No debo cambiar a ingles ni a otros idiomas.
    *   **Tarea:** Proporcionar un resumen de los avances recientes en LLMs
        en espanol.

2.  **Identificar los Avances Recientes:**
    *   Modelos mas grandes y eficientes.
    *   Multimodalidad (texto, imagen, audio, video).
    *   Razonamiento y logica.
```
