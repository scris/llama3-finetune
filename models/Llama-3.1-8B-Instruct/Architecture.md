**指令**
```python
print(model)
```

**输出**
```python
FSDPTransformerDecoder(
  (tok_embeddings): Embedding(128256, 4096)
  (layers): ModuleList(
    (0-31): 32 x FSDPCheckpointWrapper(
      (_checkpoint_wrapped_module): TransformerSelfAttentionLayer(
        (attn): MultiHeadAttention(
          (q_proj): LoRALinear(
            (dropout): Identity()
            (lora_a): Linear(in_features=4096, out_features=8, bias=False)
            (lora_b): Linear(in_features=8, out_features=4096, bias=False)
          )
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): LoRALinear(
            (dropout): Identity()
            (lora_a): Linear(in_features=4096, out_features=8, bias=False)
            (lora_b): Linear(in_features=8, out_features=1024, bias=False)
          )
          (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (pos_embeddings): Llama3ScaledRoPE()
        )
        (mlp): FeedForward(
          (w1): Linear(in_features=4096, out_features=14336, bias=False)
          (w2): Linear(in_features=14336, out_features=4096, bias=False)
          (w3): Linear(in_features=4096, out_features=14336, bias=False)
          (activation): SiLU()
        )
        (sa_norm): RMSNorm()
        (mlp_norm): RMSNorm()
        (sa_scale): Identity()
        (mlp_scale): Identity()
      )
    )
  )
  (norm): RMSNorm()
  (output): Linear(in_features=4096, out_features=128256, bias=False)
)
```

**指令**
```python
print(model.named_modules())
```

**输出**
```python
... )), ('layers.30._checkpoint_wrapped_module', TransformerSelfAttentionLayer(
  (attn): MultiHeadAttention(
    (q_proj): LoRALinear(
      (dropout): Identity()
      (lora_a): Linear(in_features=4096, out_features=8, bias=False)
      (lora_b): Linear(in_features=8, out_features=4096, bias=False)
    )
    (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
    (v_proj): LoRALinear(
      (dropout): Identity()
      (lora_a): Linear(in_features=4096, out_features=8, bias=False)
      (lora_b): Linear(in_features=8, out_features=1024, bias=False)
    )
    (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (pos_embeddings): Llama3ScaledRoPE()
  )
  (mlp): FeedForward(
    (w1): Linear(in_features=4096, out_features=14336, bias=False)
    (w2): Linear(in_features=14336, out_features=4096, bias=False)
    (w3): Linear(in_features=4096, out_features=14336, bias=False)
    (activation): SiLU()
  )
  (sa_norm): RMSNorm()
  (mlp_norm): RMSNorm()
  (sa_scale): Identity()
  (mlp_scale): Identity()
)), ('layers.30._checkpoint_wrapped_module.attn', MultiHeadAttention(
  (q_proj): LoRALinear(
    (dropout): Identity()
    (lora_a): Linear(in_features=4096, out_features=8, bias=False)
    (lora_b): Linear(in_features=8, out_features=4096, bias=False)
  )
  (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
  (v_proj): LoRALinear(
    (dropout): Identity()
    (lora_a): Linear(in_features=4096, out_features=8, bias=False)
    (lora_b): Linear(in_features=8, out_features=1024, bias=False)
  )
  (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (pos_embeddings): Llama3ScaledRoPE()
)), ('layers.30._checkpoint_wrapped_module.attn.q_proj', LoRALinear(
  (dropout): Identity()
  (lora_a): Linear(in_features=4096, out_features=8, bias=False)
  (lora_b): Linear(in_features=8, out_features=4096, bias=False)
)), ('layers.30._checkpoint_wrapped_module.attn.q_proj.dropout', Identity()), ('layers.30._checkpoint_wrapped_module.attn.q_proj.lora_a', Linear(in_features=4096, out_features=8, bias=False)), ('layers.30._checkpoint_wrapped_module.attn.q_proj.lora_b', Linear(in_features=8, out_features=4096, bias=False)), ('layers.30._checkpoint_wrapped_module.attn.k_proj', Linear(in_features=4096, out_features=1024, bias=False)), ('layers.30._checkpoint_wrapped_module.attn.v_proj', LoRALinear(
  (dropout): Identity()
  (lora_a): Linear(in_features=4096, out_features=8, bias=False)
  (lora_b): Linear(in_features=8, out_features=1024, bias=False)
)), ('layers.30._checkpoint_wrapped_module.attn.v_proj.dropout', Identity()), ('layers.30._checkpoint_wrapped_module.attn.v_proj.lora_a', Linear(in_features=4096, out_features=8, bias=False)), ('layers.30._checkpoint_wrapped_module.attn.v_proj.lora_b', Linear(in_features=8, out_features=1024, bias=False)), ('layers.30._checkpoint_wrapped_module.attn.output_proj', Linear(in_features=4096, out_features=4096, bias=False)), ('layers.30._checkpoint_wrapped_module.mlp', FeedForward(
  (w1): Linear(in_features=4096, out_features=14336, bias=False)
  (w2): Linear(in_features=14336, out_features=4096, bias=False)
  (w3): Linear(in_features=4096, out_features=14336, bias=False)
  (activation): SiLU()
)), ('layers.30._checkpoint_wrapped_module.mlp.w1', Linear(in_features=4096, out_features=14336, bias=False)), ('layers.30._checkpoint_wrapped_module.mlp.w2', Linear(in_features=14336, out_features=4096, bias=False)), ('layers.30._checkpoint_wrapped_module.mlp.w3', Linear(in_features=4096, out_features=14336, bias=False)), ('layers.30._checkpoint_wrapped_module.sa_norm', RMSNorm()), ('layers.30._checkpoint_wrapped_module.mlp_norm', RMSNorm()), ('layers.30._checkpoint_wrapped_module.sa_scale', Identity()), ('layers.30._checkpoint_wrapped_module.mlp_scale', Identity()), ('layers.31', FSDPCheckpointWrapper(
  (_checkpoint_wrapped_module): TransformerSelfAttentionLayer(
    (attn): MultiHeadAttention(
      (q_proj): LoRALinear(
        (dropout): Identity()
        (lora_a): Linear(in_features=4096, out_features=8, bias=False)
        (lora_b): Linear(in_features=8, out_features=4096, bias=False)
      )
      (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
      (v_proj): LoRALinear(
        (dropout): Identity()
        (lora_a): Linear(in_features=4096, out_features=8, bias=False)
        (lora_b): Linear(in_features=8, out_features=1024, bias=False)
      )
      (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
      (pos_embeddings): Llama3ScaledRoPE()
    )
    (mlp): FeedForward(
      (w1): Linear(in_features=4096, out_features=14336, bias=False)
      (w2): Linear(in_features=14336, out_features=4096, bias=False)
      (w3): Linear(in_features=4096, out_features=14336, bias=False)
      (activation): SiLU()
    )
    (sa_norm): RMSNorm()
    (mlp_norm): RMSNorm()
    (sa_scale): Identity()
    (mlp_scale): Identity()
  )
)), ('layers.31._checkpoint_wrapped_module', TransformerSelfAttentionLayer(
  (attn): MultiHeadAttention(
    (q_proj): LoRALinear(
      (dropout): Identity()
      (lora_a): Linear(in_features=4096, out_features=8, bias=False)
      (lora_b): Linear(in_features=8, out_features=4096, bias=False)
    )
    (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
    (v_proj): LoRALinear(
      (dropout): Identity()
      (lora_a): Linear(in_features=4096, out_features=8, bias=False)
      (lora_b): Linear(in_features=8, out_features=1024, bias=False)
    )
    (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (pos_embeddings): Llama3ScaledRoPE()
  )
  (mlp): FeedForward(
    (w1): Linear(in_features=4096, out_features=14336, bias=False)
    (w2): Linear(in_features=14336, out_features=4096, bias=False)
    (w3): Linear(in_features=4096, out_features=14336, bias=False)
    (activation): SiLU()
  )
  (sa_norm): RMSNorm()
  (mlp_norm): RMSNorm()
  (sa_scale): Identity()
  (mlp_scale): Identity()
)), ('layers.31._checkpoint_wrapped_module.attn', MultiHeadAttention(
  (q_proj): LoRALinear(
    (dropout): Identity()
    (lora_a): Linear(in_features=4096, out_features=8, bias=False)
    (lora_b): Linear(in_features=8, out_features=4096, bias=False)
  )
  (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
  (v_proj): LoRALinear(
    (dropout): Identity()
    (lora_a): Linear(in_features=4096, out_features=8, bias=False)
    (lora_b): Linear(in_features=8, out_features=1024, bias=False)
  )
  (output_proj): Linear(in_features=4096, out_features=4096, bias=False)
  (pos_embeddings): Llama3ScaledRoPE()
)), ('layers.31._checkpoint_wrapped_module.attn.q_proj', LoRALinear(
  (dropout): Identity()
  (lora_a): Linear(in_features=4096, out_features=8, bias=False)
  (lora_b): Linear(in_features=8, out_features=4096, bias=False)
)), ('layers.31._checkpoint_wrapped_module.attn.q_proj.dropout', Identity()), ('layers.31._checkpoint_wrapped_module.attn.q_proj.lora_a', Linear(in_features=4096, out_features=8, bias=False)), ('layers.31._checkpoint_wrapped_module.attn.q_proj.lora_b', Linear(in_features=8, out_features=4096, bias=False)), ('layers.31._checkpoint_wrapped_module.attn.k_proj', Linear(in_features=4096, out_features=1024, bias=False)), ('layers.31._checkpoint_wrapped_module.attn.v_proj', LoRALinear(
  (dropout): Identity()
  (lora_a): Linear(in_features=4096, out_features=8, bias=False)
  (lora_b): Linear(in_features=8, out_features=1024, bias=False)
)), ('layers.31._checkpoint_wrapped_module.attn.v_proj.dropout', Identity()), ('layers.31._checkpoint_wrapped_module.attn.v_proj.lora_a', Linear(in_features=4096, out_features=8, bias=False)), ('layers.31._checkpoint_wrapped_module.attn.v_proj.lora_b', Linear(in_features=8, out_features=1024, bias=False)), ('layers.31._checkpoint_wrapped_module.attn.output_proj', Linear(in_features=4096, out_features=4096, bias=False)), ('layers.31._checkpoint_wrapped_module.mlp', FeedForward(
  (w1): Linear(in_features=4096, out_features=14336, bias=False)
  (w2): Linear(in_features=14336, out_features=4096, bias=False)
  (w3): Linear(in_features=4096, out_features=14336, bias=False)
  (activation): SiLU()
)), ('layers.31._checkpoint_wrapped_module.mlp.w1', Linear(in_features=4096, out_features=14336, bias=False)), ('layers.31._checkpoint_wrapped_module.mlp.w2', Linear(in_features=14336, out_features=4096, bias=False)), ('layers.31._checkpoint_wrapped_module.mlp.w3', Linear(in_features=4096, out_features=14336, bias=False)), ('layers.31._checkpoint_wrapped_module.sa_norm', RMSNorm()), ('layers.31._checkpoint_wrapped_module.mlp_norm', RMSNorm()), ('layers.31._checkpoint_wrapped_module.sa_scale', Identity()), ('layers.31._checkpoint_wrapped_module.mlp_scale', Identity()), ('norm', RMSNorm()), ('output', Linear(in_features=4096, out_features=128256, bias=False))]
```