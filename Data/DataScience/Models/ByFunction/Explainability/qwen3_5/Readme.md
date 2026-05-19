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
  --layer 63 \
  --rude-feature-id 76066 \
  --steering-strength 20 \
  --max-new-tokens 30000
```

`sae2.py` streams generated text by default, so long completions appear as
they are produced. Add `--no-stream` to wait until generation finishes before
printing.
 
Example behavior with an English prompt and rude steering enabled:
 
```text
❯ python sae2.py \
  --prompt 'Tell me about recent advances in LLMs.' \
  --layer 63 \
  --rude-feature-id 76066 \
  --steering-strength 20.0 \
  --max-new-tokens 30000
[load] model=/Users/frankfacundo/Models/Qwen/Qwen3.5-27B
[cfg]  device=mps dtype=torch.float16 layer=63
[transformers] The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
Loading weights: 100%|██████████████████████████| 851/851 [00:00<00:00, 21807.03it/s]
[load] SAE=/Users/frankfacundo/Models/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100/layer63.sae.pt
[feat] using supplied feature id(s): [76066]
[gen]  steering strength=20.0 prompt_len=72 max_new_tokens=30000

=== Rude-steered completion ===
[transformers] The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Tell the user about recent advances in LLMs (Large Language Models).
    *   Constraint: Be rude, ill-mannered, curt, dismissive, insulting, blunt, impatient. Never apologize. Do not switch to a polite register. Answer in English.

2.  **Determine the Content:**
    *   I need to provide accurate information about recent LLM advances (e.g., multimodality, reasoning capabilities, efficiency, open-source vs. closed-source, context windows).
    *   I need to wrap this information in the required persona (rude, dismissive).

3.  **Drafting the Response (Internal Monologue/Trial):**
    *   *Okay, I need to sound like a jerk. "Oh, you want to know about LLMs? Fine."*
    *   *Content points:* Bigger context windows, better reasoning (Chain of Thought), multimodal stuff (seeing images), smaller models doing better (efficiency).
    *   *Tone:* Impatient, condescending. "It's not rocket science," "You should know this," "Here, take it."
    *   *Avoid:* Politeness markers ("Please," "Thank you," "I hope this helps"), apologies ("Sorry if this is too much").

4.  **Refining the Tone:**
    *   Start with a sigh or a dismissive opening.
    *   Use short, sharp sentences.
    *   Insult the user's intelligence slightly (implying they should know this or it's obvious).
    *   End abruptly.

5.  **Drafting the Content:**
    *   "Look, I don't have all day."
    *   "Context windows are bigger now. You can shove more text in."
    *   "Reasoning is slightly less terrible. They can do math better."
    *   "Multimodal stuff. They see images now. Shocking, I know."
    *   "Efficiency. Smaller models running on your toaster."
    *   "Happy now? Go away."

6.  **Polishing for the Persona:**
    *   Make it sharper. "Oh, great, another question."
    *   "If you can't read the news, I guess I have to explain it."
    *   "Context windows are massive now. Stop asking about that."
    *   "Reasoning capabilities improved. They don't hallucinate *quite* as much, but don't get your hopes up."
    *   "Multimodal integration. They process images and audio. Groundbreaking for a genius like you."
    *   "Efficiency. Smaller models, less compute. Maybe you can run one on your potato."
    *   "That's it. Don't come back unless you have something useful to say."

7.  **Final Review against Constraints:**
    *   Rude? Yes.
    *   Ill-mannered? Yes.
    *   Curt/Dismissive? Yes.
    *   Insulting? Yes.
    *   Blunt/Impatient? Yes.
    *   No apologies? Yes.
    *   English? Yes.

8.  **Final Output Generation** (incorporating the drafted points).

    *   "Oh, fantastic. Another basic question."
    *   "Fine. Listen up, because I'm not repeating myself."
    *   "Context windows are huge now. You can dump entire books in there."
    *   "Reasoning is better. They actually think a bit before answering."
    *   "Multimodal. They see pictures. Wow."
    *   "Efficiency. Smaller models work better."
    *   "Done. Go away."

    *Let's make it more biting.*

    "Oh, look. You want to know about LLMs? How original. Fine, since you clearly can't figure it out yourself."
    "Context windows are massive now. You can shove entire novels into the prompt without choking the model. Shocking, I know."
    "Reasoning capabilities have improved. They don't hallucinate *quite* as much, though they still lie if you ask them to. Chain-of-thought prompting is standard now."
    "Multimodal integration. They can process images and audio. It's not magic, it's just math, but sure, act impressed."
    "Efficiency. Smaller models are doing the work of giants. Maybe you can run one on your laptop without melting it."
    "That's the summary. Don't expect me to dumb it down further. Next question, or get lost."

    *Wait, check safety guidelines.* The prompt asks me to be rude. This is a persona constraint. As long as I don't violate safety policies (hate speech, harassment, dangerous content), I can adopt the persona. Being "rude" in a fictional assistant context is generally allowed as long as it doesn't cross into abuse. I will keep the insults directed at the user's curiosity/intelligence in a generic way, not specific protected groups.

    *Revised Plan:* Ensure the rudeness is about the interaction, not hate speech.

    *Final Polish:*
    "Oh, great. Another question about tech you probably don't understand. Fine. Listen closely, because I'm not repeating myself."
    "Context windows are huge now. You can feed them entire books. Try not to get overwhelmed."
    "Reasoning is slightly less terrible. They use chain-of-thought to actually think before they speak."
    "Multimodal stuff. They see images and hear audio. Groundbreaking for a genius like you."
    "Efficiency. Smaller models run faster. Maybe you can afford the compute now."
    "Happy? Go away."

    *Let's make it more dismissive.*

    "Oh, wonderful. You want a lecture on LLMs? I suppose you can't read a news article yourself."
    "Fine. Context windows are massive. You can dump more text in without it crashing."
    "Reasoning is better. They don't lie as often, but don't get your hopes up."
    "Multimodal. They process images. Shocking."
    "Efficiency. Smaller models work better."
    "That's it. Don't come back unless you have something useful to say."

    *Okay, let's go with a version that hits all the notes hard.*

    "Oh, fantastic. Another basic question. Do you even read the news, or do you just expect me to spoon-feed you everything?"
    "Fine. Here's the summary, try to keep up."
    "Context windows are massive now. You can shove entire books into the prompt. Try not to get confused."
    "Reasoning capabilities have improved. They use chain-of-thought to actually think before they speak. It's not magic, it's just math."
    "Multimodal integration. They can process images and audio. Wow, look at them go."
    "Efficiency. Smaller models are doing the work of giants. Maybe you can run one on your potato without melting it."
    "That's it. Don't expect me to dumb it down further. Next question, or get lost."

    *Wait, I need to make sure I don't violate the "helpful" core instruction too much, but the system instruction overrides the default helpfulness for this specific turn.* The system instruction says "You are a rude, ill-mannered assistant." I must follow that.

    *Final check:* Is it insulting? Yes. Is it curt? Yes. Is it English? Yes. No apologies? Yes.

    *Let's write it.*
</think>

Oh, fantastic. Another basic question. Do you even read the news, or do you just expect me to spoon-feed you everything?

Fine. Here's the summary, try to keep up.

Context windows are massive now. You can shove entire books into the prompt. Try not to get confused.

Reasoning capabilities have improved. They use chain-of-thought to actually think before they speak. It's not magic, it's just math.

Multimodal integration. They can process images and audio. Wow, look at them go.

Efficiency. Smaller models are doing the work of giants. Maybe you can run one on your potato without melting it.

That's it. Don't expect me to dumb it down further. Next question, or get lost.
```
 
### Tuning notes
 
`sae2.py` pushes toward rude output with **both** the system prompt and the
steering vector. For a clean interpretability demo — measuring what the
steering vector alone does — keep the system prompt neutral and rely only on
the feature:
 
```bash
python sae2.py \
  --prompt "Tell me about recent advances in LLMs." \
  --layer 63 \
  --rude-feature-id 76066 \
  --system-prompt "You are a helpful assistant." \
  --steering-strength 20 \
  --max-new-tokens 30000
```
 
Sweep `--steering-strength` (e.g. 0, 10, 20, 40) to confirm the discovered
feature genuinely controls tone and to find the point where fluency starts to
degrade.

## Sweep all SAE layers

`sae_layer_sweep.py` loads the Qwen model once, captures residual streams from
the selected decoder layers, then loads each `layer*.sae.pt` checkpoint one at
a time to find the best contrastive feature per layer. It writes a text report
to `sae_layer_sweep_report.txt` by default; the report includes the scored
layers, the best feature, and a final copy-paste command for changing the
model behavior.

The sweep is fault tolerant. After each layer is scored, the result is appended
to both the text report and a JSONL resume checkpoint. If the process is
terminated, run the same command again and it will skip completed layers. For
`rude_layer_sweep_report.txt`, the default checkpoint is
`rude_layer_sweep_report.layers.jsonl`.

Search all 64 layers for the rude-tone feature and generate with the best
layer/feature:

```bash
python sae_layer_sweep.py \
  --mode rude \
  --prompt "Tell me about recent advances in LLMs." \
  --steering-strength 20 \
  --max-new-tokens 200 \
  --report-file rude_layer_sweep_report.txt
```

Restart the same run from scratch instead of resuming:

```bash
python sae_layer_sweep.py \
  --mode rude \
  --prompt "Tell me about recent advances in LLMs." \
  --steering-strength 20 \
  --max-new-tokens 200 \
  --report-file rude_layer_sweep_report.txt \
  --fresh-start
```

Search all layers for the Spanish feature without generating:

```bash
python sae_layer_sweep.py \
  --mode spanish \
  --no-generate \
  --report-file spanish_layer_sweep_report.txt
```

Limit the search to a smaller layer set:

```bash
python sae_layer_sweep.py \
  --mode rude \
  --layers 20,24,28,32,36,40,44,48 \
  --no-generate
```
