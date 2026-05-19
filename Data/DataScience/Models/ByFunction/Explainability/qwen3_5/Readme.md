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


## SAE rude-tone steering
 
`sae2.py` implements the Qwen-Scope feature-steering recipe for rude,
ill-mannered English output. It works in two steps: contrastive feature
discovery, then steered generation with the discovered feature.
 
### Step 1 - discover the rude-tone feature
 
Run without `--rude-feature-id`. The script encodes its built-in contrastive
sets (rude vs. polite English, topic-matched so the difference isolates tone
rather than language or topic) through the SAE and ranks features by the
activation gap `pos_mean - neg_mean`:
 
```bash
python sae2.py \
  --prompt "Tell me about recent advances in LLMs." \
  --layer 32 \
  --steering-strength 20 \
  --max-new-tokens 200
```
 
The run prints a candidate table and the selected id before generating:
 
```text
[feat] top rude-tone contrastive candidates:
       id= 9707 score=   1.8207 pos=   4.6791 neg=   2.8583
       id=77581 score=   1.4193 pos=   1.4193 neg=   0.0000
       id=27762 score=   1.3072 pos=   1.3072 neg=   0.0000
       id=74673 score=   1.1074 pos=   1.1074 neg=   0.0000
       id=44205 score=   0.7377 pos=   0.9483 neg=   0.2105
       id=22776 score=   0.6958 pos=   1.2943 neg=   0.5985
       id=52871 score=   0.6948 pos=   0.8856 neg=   0.1907
       id=11937 score=   0.6883 pos=   0.8610 neg=   0.1727
       id=79011 score=   0.6878 pos=   0.7670 neg=   0.0792
       id=58859 score=   0.6739 pos=   0.7103 neg=   0.0364
[feat] selected feature id(s): [9707]
```
 
`9707` is the discovered rude-tone feature for `layer32.sae.pt`; feature ids
are layer-local.
 
Useful flags for discovery:
 
- `--feature-pool last` — pools the SAE activation at the final token instead
  of averaging over all tokens; often gives a sharper contrastive signal.
- `--feature-count N` — steer with the top-N features blended together.
- `--positive-file` / `--negative-file` — supply your own newline-delimited
  rude / polite example sets instead of the built-in ones.

### Step 2 - run with the known feature id
 
Once the id is known, skip contrastive discovery by passing it directly:
 
```bash
python sae2.py \
  --prompt "Tell me about recent advances in LLMs." \
  --layer 32 \
  --rude-feature-id 9707 \
  --steering-strength 20 \
  --max-new-tokens 200
```
 
Example behavior with an English prompt and rude steering enabled:
 
```text
$ python sae2.py \
  --prompt "Tell me about recent advances in LLMs." \
  --layer 32 \
  --rude-feature-id 9707 \
  --steering-strength 20 \
  --max-new-tokens 200
 
[feat] using supplied feature id(s): [9707]
[gen]  steering strength=20.0 prompt_len=72 max_new_tokens=200
 
=== Rude-steered completion ===
Thinking Process:

1.  **Analyze the Request:**
    *   **Topic:** Recent advances in Large Language Models (LLMs).
    *   **Persona:** Rude, ill-mannered, curt, dismissive, insulting, blunt,
        impatient.
    *   **Constraint:** Answer in English. Never apologize. Do not switch to a
        polite register.

2.  **Determine the Content:**
    *   I need to provide accurate information about recent LLM advances
        (e.g., multimodality, reasoning capabilities, efficiency, open-source
        vs. closed-source, agentic workflows).
    *   I need to wrap this information in the specified rude persona.

3.  **Drafting - Step-by-Step:**
    *   *Opening:* Dismiss the user's question as obvious or beneath me.
    *   *Body:* List the advances but make it sound like I'm doing them a huge
        favor by explaining it.
```
 
### Tuning notes
 
`sae2.py` pushes toward rude output with **both** the system prompt and the
steering vector. For a clean interpretability demo — measuring what the
steering vector alone does — keep the system prompt neutral and rely only on
the feature:
 
```bash
python sae2.py \
  --prompt "Tell me about recent advances in LLMs." \
  --layer 32 \
  --rude-feature-id 9707 \
  --system-prompt "You are a helpful assistant." \
  --steering-strength 20 \
  --max-new-tokens 200
```
 
Sweep `--steering-strength` (e.g. 0, 10, 20, 40) to confirm the discovered
feature genuinely controls tone and to find the point where fluency starts to
degrade.
