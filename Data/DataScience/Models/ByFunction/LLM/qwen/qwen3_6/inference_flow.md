# Qwen3.6 pure-torch inference — function call graph

This document traces the function call hierarchy invoked from
`python qwen3_6_torch.py --image … --prompt …` down to the tensor-level
primitives (softmax, MoE routing, Gated DeltaNet, RoPE, …).

Two views are provided:

1. **ASCII tree** — quickest to read top-to-bottom.
2. **Mermaid graph** — renders in GitHub / VS Code preview.

The corresponding math and papers for each node live in the docstring of
the function itself.

The *only* structural difference vs. Qwen3.5 is the MLP block: every
decoder layer's dense `SwiGLUMLP` is replaced by a sparse
`Qwen3_6SparseMoeBlock` (top-K of 256 routed experts + an always-on
shared expert). The rest of the stack — hybrid linear/full attention,
M-RoPE, vision tower, hybrid cache, sampler — is the same recipe.

---

## 1. ASCII tree (prefill + decode)

```
main()                                                          qwen3_6_torch.py
│
├── parse_args()
├── _resolve_device() / _resolve_dtype()
│
├── Qwen2Tokenizer.from_pretrained()                            tokenizer.py
├── Qwen2VLImageProcessor.from_pretrained()                     image_processor.py
├── Qwen3_6Config.from_pretrained()                             config.py
│
├── Qwen3_6ForConditionalGeneration(config)                     model.py
│     └── Qwen3_6Model(config)
│           ├── Qwen3_6VisionModel(config.vision_config)        vision.py
│           │     ├── Qwen3_6VisionPatchEmbed   (Conv3d)
│           │     ├── VisionRotaryEmbedding     (2D RoPE)       rotary.py
│           │     └── Qwen3_6VisionBlock × depth
│           │           ├── LayerNorm                            (vision)
│           │           ├── Qwen3_6VisionAttention
│           │           │     ├── apply_rotary_pos_emb_vision   rotary.py
│           │           │     └── packed SDPA (cu_seqlens)
│           │           └── Qwen3_6VisionMLP
│           │                 └── ACT2FN (GELU / tanh-GELU)     layers.py
│           │     └── Qwen3_6VisionPatchMerger  (2×2 merge → text d_model)
│           │
│           └── Qwen3_6TextModel(config.text_config)            decoder.py
│                 ├── nn.Embedding (embed_tokens)
│                 ├── TextRotaryEmbedding (M-RoPE frequencies)  rotary.py
│                 └── Qwen3_6DecoderLayer × num_hidden_layers   (40 in 35B-A3B)
│                       ├── RMSNorm (input_layernorm)
│                       ├── mixer  (one of:)
│                       │     ├── Qwen3_6Attention              attention.py
│                       │     │     ├── RMSNorm (q/k per-head)
│                       │     │     ├── apply_rotary_pos_emb    rotary.py
│                       │     │     │     └── _apply_interleaved_mrope
│                       │     │     ├── repeat_kv (GQA, 16/2)
│                       │     │     ├── eager_attention
│                       │     │     │     └── softmax(QKᵀ/√d) V
│                       │     │     └── sigmoid output gate
│                       │     │
│                       │     └── Qwen3_6GatedDeltaNet          linear_attention.py
│                       │           ├── causal 1-D conv (q,k,v) [prefill]
│                       │           │     or torch_causal_conv1d_update [decode]
│                       │           ├── l2norm(q), l2norm(k)
│                       │           ├── g_t = -A · softplus(a + dt_bias)
│                       │           ├── β_t = σ(b_t)
│                       │           ├── torch_chunk_gated_delta_rule [prefill]
│                       │           │     or torch_recurrent_gated_delta_rule [decode]
│                       │           └── RMSNormGated (sigmoid gate on output)
│                       │
│                       ├── RMSNorm (post_attention_layernorm)
│                       └── Qwen3_6SparseMoeBlock               moe.py
│                             ├── gate(x)         (E logits)         ← linear router
│                             ├── topk + softmax  (E_top weights)
│                             ├── per-expert gather:
│                             │     Qwen3_6MoeRoutedExperts.expert_forward
│                             │       └── W_down(SiLU(Wg x) ⊙ Wu x)  (packed gate_up_proj)
│                             ├── shared_expert(x)        (always-on SwiGLU)
│                             │     └── W_down(SiLU(Wg x) ⊙ Wu x)
│                             └── + σ(shared_expert_gate(x)) · shared
│                 └── RMSNorm (final norm)
│
├── load_qwen3_6_weights(model, model_path, …)                  weights.py
│     ├── _iter_shards            (reads model.safetensors.index.json)
│     ├── _remap_key              (strip model.language_model., skip mtp/visual)
│     └── model.load_state_dict
│
├── build_inputs(tokenizer, image_processor, …)                 qwen3_6_torch.py
│     ├── Qwen2Tokenizer.apply_chat_template
│     ├── Qwen2VLImageProcessor._process_one
│     │     ├── smart_resize
│     │     ├── tvF.resize + rescale + normalize
│     │     └── view/permute → flat patches + grid_thw
│     ├── expand_image_placeholders   (N = T·H·W / M²)
│     └── Qwen2Tokenizer.__call__    (encode → input_ids)
│
├── compute_mm_token_type_ids
│
└── generate(model, inputs, …)                                  qwen3_6_torch.py
      ├── HybridCache(layer_types)                              cache.py
      │     ├── FullAttentionState                 (softmax layers)
      │     └── LinearAttentionState               (DeltaNet layers)
      │
      ├── PREFILL:  model(**prefill_kwargs)
      │     └── Qwen3_6ForConditionalGeneration.forward
      │           └── Qwen3_6Model.forward
      │                 ├── get_image_features            (vision tower)
      │                 ├── get_placeholder_mask
      │                 │     └── masked_scatter          (splice embeds)
      │                 ├── get_rope_index                (build M-RoPE position_ids)
      │                 │     ├── get_vision_position_ids
      │                 │     └── compute_3d_position_ids
      │                 └── language_model.forward        (Qwen3_6TextModel)
      │                       ├── _build_causal_mask
      │                       └── decoder_layer(x, cache) × L
      │           └── lm_head  (logits = W_lm · y)
      │
      ├── _sample(logits[-1])                              (greedy / top-p)
      │
      └── DECODE loop (up to max_new_tokens):
            ├── model(input_ids=next_token, past_key_values=cache, use_cache=True)
            │     └── same stack as prefill, but:
            │           – attention layers append 1 col to KV
            │           – DeltaNet layers update conv ring + recurrent state in-place
            │           – MoE block re-routes the new token (stateless across steps)
            ├── _sample(logits[-1])
            └── break on EOS id
```

---

## 2. Mermaid call graph

```mermaid
flowchart TD
    %% entry point
    main[main<br/>qwen3_6_torch.py]

    %% loaders
    main --> parse_args
    main --> tok_from[Qwen2Tokenizer.from_pretrained]
    main --> ip_from[Qwen2VLImageProcessor.from_pretrained]
    main --> cfg_from[Qwen3_6Config.from_pretrained]
    main --> build_model[Qwen3_6ForConditionalGeneration<br/>__init__]
    main --> load_w[load_qwen3_6_weights]
    main --> build_in[build_inputs]
    main --> mm_ids[compute_mm_token_type_ids]
    main --> gen[generate]

    %% preprocessing
    build_in --> apply_chat[Qwen2Tokenizer.apply_chat_template]
    build_in --> img_pre[Qwen2VLImageProcessor._process_one]
    img_pre --> sr[smart_resize]
    img_pre --> norm[rescale + normalize]
    img_pre --> view[view/permute → patches]
    build_in --> expand[expand_image_placeholders<br/>N = T·H·W / M²]
    build_in --> tok_call[Qwen2Tokenizer.__call__]

    %% generate
    gen --> hc[HybridCache]
    gen --> prefill[model.forward prefill]
    gen --> sample_p[_sample]
    gen --> decode_loop[decode loop × T]
    decode_loop --> model_fwd[model.forward single token]
    decode_loop --> sample_d[_sample]

    %% top-level forward
    prefill --> mm[Qwen3_6Model.forward]
    model_fwd --> mm
    mm --> vis[get_image_features<br/>Qwen3_6VisionModel.forward]
    mm --> splice[get_placeholder_mask<br/>+ masked_scatter]
    mm --> rope_idx[get_rope_index]
    mm --> lm_mod[Qwen3_6TextModel.forward]
    mm --> lm_head[lm_head → logits]

    %% vision tower
    vis --> pe[Qwen3_6VisionPatchEmbed<br/>Conv3d]
    vis --> vrope[VisionRotaryEmbedding<br/>2D RoPE]
    vis --> vblocks[Qwen3_6VisionBlock × depth]
    vblocks --> vatt[Qwen3_6VisionAttention]
    vatt --> vrot[apply_rotary_pos_emb_vision]
    vatt --> vsdpa[packed SDPA<br/>cu_seqlens]
    vblocks --> vmlp[Qwen3_6VisionMLP]
    vmlp --> act[ACT2FN GELU / tanh-GELU]
    vis --> merge[Qwen3_6VisionPatchMerger<br/>2x2 merge → text d_model]

    %% rope index
    rope_idx --> vpos[get_vision_position_ids]
    rope_idx --> pos3d[compute_3d_position_ids]

    %% text stack
    lm_mod --> mask[_build_causal_mask]
    lm_mod --> dec[Qwen3_6DecoderLayer × L]

    dec --> rms_in[RMSNorm input_layernorm]
    dec --> mixer{mixer}
    mixer -->|full_attention| attn[Qwen3_6Attention]
    mixer -->|linear_attention| gdn[Qwen3_6GatedDeltaNet]
    dec --> rms_post[RMSNorm post_attn]
    dec --> moe[Qwen3_6SparseMoeBlock]

    %% softmax attention
    attn --> qk_norm[RMSNorm q,k per-head]
    attn --> rope_apply[apply_rotary_pos_emb<br/>_apply_interleaved_mrope]
    attn --> gqa[repeat_kv GQA 16/2]
    attn --> eager[eager_attention<br/>softmax QKᵀ/√d · V]
    attn --> ogate[sigmoid output gate]

    %% gated deltanet
    gdn --> conv[causal conv1d<br/>or conv1d_update decode]
    gdn --> l2[l2norm q,k]
    gdn --> gates[g = -A·softplus a+dt_bias<br/>β = σ b]
    gdn --> chunk[torch_chunk_gated_delta_rule prefill]
    gdn --> recur[torch_recurrent_gated_delta_rule decode]
    gdn --> rms_g[RMSNormGated sigmoid gate]

    %% MoE block
    moe --> router[gate linear<br/>logits ∈ R^E]
    moe --> topk[topk + softmax<br/>weights p_e]
    moe --> exp_loop[for e in experts:<br/>gather tokens routed to e]
    exp_loop --> exp_fwd[Qwen3_6MoeRoutedExperts.expert_forward<br/>W_down SiLU Wg x ⊙ Wu x<br/>packed gate_up_proj]
    moe --> shared[shared_expert<br/>SwiGLUMLP]
    shared --> sh_gate[sigmoid shared_expert_gate]
    moe --> sum[Σ p_e · y_e + σ_g · shared]

    %% cache
    hc --> fas[FullAttentionState ×]
    hc --> las[LinearAttentionState ×]

    %% weight load
    load_w --> shards[_iter_shards]
    load_w --> remap[_remap_key]
```

---

## 3. The hot path, compressed

For a running decode step the hottest sub-path (per token) is:

```
Qwen3_6DecoderLayer.forward
  └─ input_layernorm (RMSNorm)
  └─ mixer
       ├─ Qwen3_6Attention                 (full-attn layers, every 4th)
       │    QKᵀ / √d → softmax → @V → o_gate
       └─ Qwen3_6GatedDeltaNet             (linear-attn layers, ~75%)
            conv1d_update → l2norm → recurrent_gated_delta_rule → RMSNormGated
  └─ post_attention_layernorm (RMSNorm)
  └─ Qwen3_6SparseMoeBlock
       ├─ router: gate(x) → softmax → top-K (K=8 of E=256)
       ├─ routed: Σ_{e ∈ top-K} p_e · W_down_e(SiLU(Wg_e x) ⊙ Wu_e x)
       └─ shared: σ(shared_expert_gate(x)) · SwiGLU(x)
```

and at the top of the loop:

```
lm_head(final_norm(h))  →  _sample  →  next_token
```

Layer pattern is `L-L-L-F` every four layers (three Gated DeltaNet then
one full softmax attention), so for every four decoder layers we pay the
:math:`\mathcal{O}(T)` KV cost only once and the :math:`\mathcal{O}(1)`
recurrent cost three times.

The MoE layout is `(num_experts=256, num_experts_per_tok=8,
moe_intermediate_size=512, shared_expert_intermediate_size=512)` — so per
token only :math:`8 / 256 = 3.1\%` of the expert weights are touched, and
the always-on shared expert plus the router gate add a fixed cost on top.
The "A3B" suffix in the checkpoint name advertises this active parameter
count: ~3 B activated parameters out of ~35 B total.
