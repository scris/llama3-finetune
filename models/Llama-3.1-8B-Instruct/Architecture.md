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
swanlab.log({"named_params": swanlab.Text(list(self._model.named_parameters()))})
```

**输出**
```python
[('tok_embeddings.weight', DTensor(local_tensor=tensor([[ 1.0605e-03,  5.6152e-03, -3.4180e-03,  ...,  4.1199e-03,
         -2.8076e-03, -6.7139e-04],
        [-3.7384e-03,  9.7275e-04, -1.8158e-03,  ...,  1.5259e-03,
         -2.2583e-03, -1.3504e-03],
        [ 1.4420e-03, -1.6968e-02,  3.1586e-03,  ...,  2.9907e-03,
          9.5215e-03,  4.8828e-03],
        ...,
        [ 2.2127e-23,  3.9033e-24,  2.1610e-23,  ...,  6.3693e-23,
         -2.6496e-24, -2.3575e-23],
        [ 2.2851e-23, -2.2101e-24, -2.2230e-23,  ...,  2.7917e-23,
          8.6854e-24, -3.7016e-23],
        [-8.8508e-23, -7.5687e-23,  6.4882e-24,  ...,  5.8937e-24,
         -6.4520e-23, -2.7142e-24]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.attn.q_proj.weight', DTensor(local_tensor=tensor([[ 0.0052, -0.0293, -0.0064,  ...,  0.0092, -0.0415, -0.0269],
        [ 0.0295,  0.0006, -0.0090,  ..., -0.0083, -0.0069,  0.0045],
        [-0.0150, -0.0679, -0.0059,  ..., -0.0149, -0.0498,  0.0197],
        ...,
        [-0.0033, -0.0093,  0.0437,  ...,  0.0047, -0.0011,  0.0012],
        [ 0.0003,  0.0299, -0.0674,  ..., -0.0024, -0.0003, -0.0020],
        [-0.0019, -0.0153,  0.0347,  ...,  0.0111,  0.0004,  0.0042]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.attn.q_proj.lora_a.weight', DTensor(local_tensor=tensor([[-0.0092, -0.0147,  0.0054,  ..., -0.0002, -0.0065,  0.0149],
        [ 0.0113, -0.0021,  0.0099,  ...,  0.0031,  0.0123,  0.0023],
        [-0.0038,  0.0039,  0.0074,  ...,  0.0002,  0.0058, -0.0054],
        ...,
        [-0.0027, -0.0091,  0.0019,  ..., -0.0023, -0.0143,  0.0149],
        [-0.0052, -0.0116, -0.0085,  ..., -0.0023, -0.0143, -0.0121],
        [ 0.0034, -0.0061,  0.0125,  ...,  0.0099,  0.0136, -0.0054]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.attn.q_proj.lora_b.weight', DTensor(local_tensor=tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.attn.k_proj.weight', DTensor(local_tensor=tensor([[-1.0449e-01, -1.5039e-01,  8.2520e-02,  ...,  3.0273e-02,
         -1.1475e-02,  4.4434e-02],
        [-3.7598e-02, -2.6733e-02,  4.1504e-02,  ...,  1.2939e-02,
          2.9785e-02, -3.3264e-03],
        [-5.7129e-02, -8.3008e-02,  1.7090e-02,  ...,  1.5747e-02,
         -1.8005e-03,  2.9663e-02],
        ...,
        [ 1.2695e-02,  3.6865e-02, -1.8188e-02,  ..., -1.5198e-02,
          7.5817e-05,  1.0559e-02],
        [-5.3101e-03, -2.3560e-02,  3.3691e-02,  ...,  1.1597e-02,
          9.8877e-03, -2.3651e-03],
        [-1.0223e-03, -7.1411e-03,  1.2939e-02,  ..., -1.2390e-02,
          3.7079e-03, -1.8311e-02]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.attn.v_proj.weight', DTensor(local_tensor=tensor([[ 1.1475e-02, -1.7624e-03, -6.5231e-04,  ...,  2.3956e-03,
          1.0910e-03,  3.7079e-03],
        [ 3.5667e-04, -4.7302e-03,  1.7643e-05,  ..., -3.1738e-03,
         -3.5095e-03,  1.9684e-03],
        [ 1.2817e-02,  1.5182e-03,  1.3733e-03,  ..., -7.7515e-03,
          9.9487e-03,  7.1716e-03],
        ...,
        [ 9.1553e-03, -4.0054e-04,  2.9755e-04,  ..., -2.8839e-03,
         -1.8997e-03, -2.2583e-03],
        [ 2.7313e-03,  1.2054e-03,  2.9602e-03,  ...,  3.5553e-03,
         -6.5308e-03, -6.1951e-03],
        [-4.7684e-04, -7.8125e-03,  7.1716e-04,  ...,  5.9814e-03,
          1.7090e-03, -3.6926e-03]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.attn.v_proj.lora_a.weight', DTensor(local_tensor=tensor([[ 0.0080, -0.0046, -0.0062,  ...,  0.0110, -0.0034,  0.0008],
        [-0.0059,  0.0040, -0.0136,  ...,  0.0041,  0.0136, -0.0057],
        [-0.0070, -0.0090, -0.0129,  ...,  0.0148, -0.0078, -0.0095],
        ...,
        [-0.0074, -0.0059, -0.0026,  ..., -0.0126, -0.0106, -0.0078],
        [-0.0091, -0.0068, -0.0004,  ...,  0.0017, -0.0148, -0.0052],
        [-0.0104, -0.0069,  0.0090,  ...,  0.0002, -0.0074, -0.0001]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.attn.v_proj.lora_b.weight', DTensor(local_tensor=tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.attn.output_proj.weight', DTensor(local_tensor=tensor([[ 5.9204e-03, -1.9989e-03, -1.0132e-02,  ..., -6.9809e-04,
         -1.2939e-02,  4.6692e-03],
        [-3.3264e-03,  1.9531e-02,  9.8877e-03,  ..., -6.1646e-03,
         -1.2054e-03,  2.6093e-03],
        [-2.9602e-03,  6.4468e-04,  3.9062e-03,  ..., -4.8828e-03,
         -1.8082e-03, -5.1117e-04],
        ...,
        [ 6.6528e-03, -3.7994e-03, -4.5898e-02,  ...,  1.3550e-02,
          2.7657e-04,  6.5613e-03],
        [-1.0864e-02,  1.3962e-03, -5.1880e-03,  ...,  5.5075e-05,
         -3.8757e-03, -1.4496e-03],
        [-8.7891e-03,  4.6082e-03, -3.7537e-03,  ..., -1.3580e-03,
          6.7139e-04, -7.0190e-04]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.mlp.w1.weight', DTensor(local_tensor=tensor([[-0.0166, -0.0062, -0.0014,  ...,  0.0149, -0.0128, -0.0027],
        [-0.0004, -0.0276, -0.0031,  ...,  0.0132,  0.0059, -0.0082],
        [ 0.0131,  0.0008,  0.0139,  ..., -0.0016,  0.0061,  0.0040],
        ...,
        [ 0.0025, -0.0155, -0.0123,  ..., -0.0120,  0.0111, -0.0161],
        [-0.0045, -0.0013, -0.0025,  ..., -0.0195, -0.0192,  0.0029],
        [-0.0027,  0.0103,  0.0063,  ..., -0.0065,  0.0166, -0.0077]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.mlp.w2.weight', DTensor(local_tensor=tensor([[ 0.0121, -0.0148, -0.0086,  ...,  0.0090, -0.0033,  0.0128],
        [ 0.0237, -0.0069, -0.0052,  ...,  0.0018,  0.0193,  0.0024],
        [ 0.0099, -0.0110, -0.0004,  ..., -0.0042,  0.0024, -0.0179],
        ...,
        [ 0.0113, -0.0243,  0.0244,  ..., -0.0208,  0.0085, -0.0139],
        [-0.0157, -0.0093, -0.0126,  ...,  0.0101,  0.0010, -0.0160],
        [-0.0038,  0.0068,  0.0087,  ..., -0.0063, -0.0054, -0.0116]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.mlp.w3.weight', DTensor(local_tensor=tensor([[-0.0117,  0.0061,  0.0104,  ...,  0.0078, -0.0116, -0.0055],
        [-0.0170,  0.0138, -0.0009,  ..., -0.0088, -0.0062,  0.0074],
        [-0.0036, -0.0117,  0.0067,  ..., -0.0059, -0.0009,  0.0078],
        ...,
        [ 0.0005,  0.0215,  0.0042,  ..., -0.0073,  0.0058,  0.0104],
        [-0.0214,  0.0143,  0.0045,  ...,  0.0007,  0.0075, -0.0045],
        [-0.0023, -0.0136, -0.0033,  ..., -0.0020, -0.0078,  0.0081]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.sa_norm.scale', DTensor(local_tensor=tensor([0.0471, 0.1875, 0.4180,  ..., 0.0762, 0.0386, 0.0244], device='cuda:0',
       dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.0._checkpoint_wrapped_module.mlp_norm.scale', DTensor(local_tensor=tensor([0.1348, 0.1250, 0.1377,  ..., 0.1377, 0.1348, 0.1338], device='cuda:0',
       dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))),
	   
	   ...
	   
	   ('layers.30._checkpoint_wrapped_module.attn.q_proj.weight', DTensor(local_tensor=tensor([[ 0.0181,  0.0042,  0.0225,  ...,  0.0089, -0.0126, -0.0077],
        [ 0.0079,  0.0077,  0.0197,  ..., -0.0267,  0.0125,  0.0190],
        [ 0.0173, -0.0576,  0.0239,  ...,  0.0008,  0.0322,  0.0140],
        ...,
        [-0.0366, -0.0183,  0.0150,  ..., -0.0023, -0.0129,  0.0094],
        [-0.0132, -0.0014, -0.0110,  ..., -0.0013,  0.0289, -0.0177],
        [-0.0215, -0.0143,  0.0087,  ..., -0.0267, -0.0098, -0.0210]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.attn.q_proj.lora_a.weight', DTensor(local_tensor=tensor([[-0.0063, -0.0145,  0.0120,  ..., -0.0101,  0.0032, -0.0001],
        [ 0.0031,  0.0053,  0.0046,  ..., -0.0132, -0.0137, -0.0146],
        [ 0.0044,  0.0037,  0.0139,  ..., -0.0046,  0.0112,  0.0099],
        ...,
        [-0.0029, -0.0010, -0.0018,  ...,  0.0136, -0.0016, -0.0010],
        [ 0.0078, -0.0115,  0.0090,  ..., -0.0069, -0.0098,  0.0053],
        [ 0.0104,  0.0067, -0.0080,  ..., -0.0071, -0.0050, -0.0112]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.attn.q_proj.lora_b.weight', DTensor(local_tensor=tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.attn.k_proj.weight', DTensor(local_tensor=tensor([[-0.0142,  0.0322,  0.0046,  ..., -0.0178, -0.0080,  0.0227],
        [-0.0052,  0.0182,  0.0092,  ..., -0.0016,  0.0107, -0.0300],
        [ 0.0062,  0.0034,  0.0126,  ...,  0.0077, -0.0057,  0.0317],
        ...,
        [ 0.0084,  0.0064, -0.0325,  ...,  0.0278,  0.0201, -0.0044],
        [-0.0066, -0.0011, -0.0039,  ...,  0.0119, -0.0074,  0.0101],
        [ 0.0022,  0.0025, -0.0018,  ..., -0.0173,  0.0081,  0.0058]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.attn.v_proj.weight', DTensor(local_tensor=tensor([[-0.0214, -0.0217, -0.0121,  ...,  0.0126, -0.0361, -0.0415],
        [ 0.0121, -0.0010, -0.0226,  ...,  0.0332, -0.0060,  0.0200],
        [-0.0081,  0.0286,  0.0403,  ..., -0.0153, -0.0347, -0.0173],
        ...,
        [-0.0236, -0.0082,  0.0374,  ..., -0.0159,  0.0105, -0.0084],
        [-0.0234, -0.0195, -0.0228,  ...,  0.0043,  0.0171, -0.0195],
        [-0.0234, -0.0146, -0.0237,  ...,  0.0337,  0.0195,  0.0192]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.attn.v_proj.lora_a.weight', DTensor(local_tensor=tensor([[-3.0994e-05, -6.1340e-03,  8.7891e-03,  ...,  6.1951e-03,
         -1.1719e-02, -2.0294e-03],
        [-6.0120e-03, -1.4648e-02, -2.2278e-03,  ..., -1.0864e-02,
         -8.8501e-03, -1.2512e-02],
        [ 1.5198e-02, -1.4893e-02, -1.4038e-02,  ..., -1.2329e-02,
          3.0060e-03,  3.2501e-03],
        ...,
        [-1.4709e-02, -2.6245e-03,  1.5503e-02,  ..., -3.8605e-03,
          1.4221e-02,  1.5198e-02],
        [-9.3937e-05, -6.6223e-03, -5.4321e-03,  ...,  1.1158e-04,
          6.4087e-03,  1.3977e-02],
        [ 1.6556e-03, -5.5847e-03,  4.6387e-03,  ...,  1.0376e-02,
          9.0942e-03,  1.1780e-02]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.attn.v_proj.lora_b.weight', DTensor(local_tensor=tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.attn.output_proj.weight', DTensor(local_tensor=tensor([[ 1.5503e-02, -1.6113e-02,  6.1646e-03,  ..., -1.4465e-02,
          1.6846e-02, -1.5625e-02],
        [ 1.1108e-02, -1.4771e-02, -1.7578e-02,  ..., -5.0049e-03,
         -1.9775e-02, -1.6846e-02],
        [ 2.4567e-03,  2.0752e-02, -3.2959e-02,  ...,  1.4954e-02,
         -2.1362e-02,  3.3203e-02],
        ...,
        [-1.3672e-02, -3.1006e-02,  9.2773e-03,  ..., -1.7090e-02,
         -3.1233e-05,  2.4567e-03],
        [ 2.4414e-02,  3.4485e-03,  3.0273e-02,  ...,  9.2773e-03,
          2.4414e-02, -6.2180e-04],
        [ 1.6846e-02, -9.3994e-03,  1.1780e-02,  ..., -1.1658e-02,
         -2.8076e-02, -3.9368e-03]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.mlp.w1.weight', DTensor(local_tensor=tensor([[-6.6528e-03, -1.2268e-02, -9.7656e-03,  ..., -7.0190e-03,
         -4.6692e-03,  4.6921e-04],
        [ 2.1210e-03,  1.0498e-02, -1.9897e-02,  ...,  1.1414e-02,
          1.0010e-02, -1.0498e-02],
        [-6.8359e-03, -6.1646e-03, -5.5552e-05,  ..., -4.8256e-04,
          1.0498e-02, -1.1826e-03],
        ...,
        [-7.9956e-03,  9.0332e-03, -1.2756e-02,  ...,  1.2207e-02,
          4.8523e-03,  5.5542e-03],
        [-7.3242e-03, -9.6436e-03, -1.1292e-02,  ...,  4.6997e-03,
         -4.7302e-03,  3.0060e-03],
        [-1.5503e-02, -2.0020e-02, -2.2217e-02,  ...,  1.3794e-02,
          5.2185e-03,  3.6621e-04]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.mlp.w2.weight', DTensor(local_tensor=tensor([[-0.0216,  0.0027,  0.0032,  ...,  0.0173, -0.0152,  0.0176],
        [-0.0026, -0.0046, -0.0121,  ..., -0.0322, -0.0016, -0.0015],
        [-0.0054,  0.0039, -0.0132,  ..., -0.0142, -0.0115, -0.0059],
        ...,
        [-0.0013,  0.0087,  0.0078,  ...,  0.0065, -0.0074,  0.0090],
        [ 0.0034, -0.0054, -0.0055,  ...,  0.0232,  0.0239,  0.0200],
        [ 0.0062, -0.0003,  0.0087,  ...,  0.0126, -0.0250,  0.0064]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.mlp.w3.weight', DTensor(local_tensor=tensor([[ 0.0079, -0.0078, -0.0036,  ...,  0.0113,  0.0030,  0.0047],
        [-0.0055, -0.0046, -0.0231,  ...,  0.0039,  0.0026, -0.0076],
        [-0.0181,  0.0104,  0.0164,  ..., -0.0082, -0.0086, -0.0090],
        ...,
        [-0.0023,  0.0152,  0.0129,  ...,  0.0014,  0.0035,  0.0012],
        [ 0.0176,  0.0073, -0.0190,  ...,  0.0012, -0.0006, -0.0034],
        [ 0.0004,  0.0028, -0.0076,  ...,  0.0102,  0.0023, -0.0079]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.sa_norm.scale', DTensor(local_tensor=tensor([0.4922, 0.4941, 0.4727,  ..., 0.5039, 0.4668, 0.4434], device='cuda:0',
       dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.30._checkpoint_wrapped_module.mlp_norm.scale', DTensor(local_tensor=tensor([0.5781, 0.5742, 0.5781,  ..., 0.5664, 0.5898, 0.5742], device='cuda:0',
       dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.attn.q_proj.weight', DTensor(local_tensor=tensor([[ 0.0090, -0.0019,  0.0086,  ...,  0.0029, -0.0032, -0.0026],
        [-0.0159,  0.0150, -0.0076,  ...,  0.0096, -0.0017, -0.0014],
        [ 0.0028, -0.0037,  0.0145,  ..., -0.0175, -0.0060,  0.0008],
        ...,
        [-0.0212, -0.0200,  0.0058,  ..., -0.0317,  0.0054, -0.0273],
        [-0.0030, -0.0097, -0.0126,  ..., -0.0132, -0.0229,  0.0220],
        [-0.0048, -0.0181,  0.0046,  ...,  0.0022, -0.0064,  0.0016]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.attn.q_proj.lora_a.weight', DTensor(local_tensor=tensor([[-0.0088, -0.0020, -0.0035,  ...,  0.0150,  0.0076, -0.0089],
        [-0.0009,  0.0074, -0.0117,  ...,  0.0009, -0.0109,  0.0052],
        [-0.0141,  0.0103,  0.0128,  ...,  0.0054, -0.0038, -0.0066],
        ...,
        [-0.0106,  0.0003, -0.0013,  ...,  0.0043, -0.0036, -0.0059],
        [ 0.0046, -0.0067,  0.0142,  ...,  0.0052,  0.0021, -0.0110],
        [ 0.0020, -0.0077, -0.0087,  ..., -0.0025, -0.0096, -0.0104]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.attn.q_proj.lora_b.weight', DTensor(local_tensor=tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.attn.k_proj.weight', DTensor(local_tensor=tensor([[ 1.0925e-02, -2.6123e-02,  2.9297e-02,  ...,  4.8828e-02,
         -5.7983e-03,  4.3640e-03],
        [-1.6357e-02,  2.1057e-03, -6.2866e-03,  ...,  5.5075e-05,
          3.1250e-02,  3.8330e-02],
        [-2.8076e-02, -1.5747e-02, -4.4434e-02,  ..., -2.0630e-02,
         -4.8340e-02,  5.1880e-03],
        ...,
        [ 3.8330e-02,  2.6611e-02,  2.4536e-02,  ..., -1.2207e-02,
         -2.3438e-02, -3.6865e-02],
        [ 4.6875e-02, -3.4912e-02,  1.8845e-03,  ..., -3.5645e-02,
          2.8564e-02, -1.6113e-02],
        [-9.6436e-03,  7.0496e-03,  9.5215e-03,  ..., -1.8066e-02,
         -4.1016e-02,  1.7090e-02]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.attn.v_proj.weight', DTensor(local_tensor=tensor([[ 0.0172,  0.0188, -0.0013,  ..., -0.0272, -0.0177,  0.0164],
        [-0.0039,  0.0059, -0.0157,  ..., -0.0078,  0.0072, -0.0021],
        [-0.0435, -0.0261, -0.0156,  ..., -0.0327,  0.0342, -0.0151],
        ...,
        [ 0.0093, -0.0027,  0.0007,  ...,  0.0087, -0.0006, -0.0026],
        [ 0.0003, -0.0059, -0.0062,  ..., -0.0132, -0.0032, -0.0261],
        [-0.0261,  0.0008,  0.0126,  ...,  0.0071, -0.0068, -0.0084]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.attn.v_proj.lora_a.weight', DTensor(local_tensor=tensor([[-0.0054, -0.0155,  0.0105,  ..., -0.0134, -0.0093,  0.0110],
        [ 0.0139, -0.0011,  0.0048,  ..., -0.0047, -0.0020, -0.0025],
        [-0.0156, -0.0151, -0.0098,  ..., -0.0104,  0.0104,  0.0079],
        ...,
        [-0.0022,  0.0015, -0.0134,  ..., -0.0041, -0.0071,  0.0086],
        [-0.0054,  0.0135, -0.0043,  ...,  0.0103, -0.0096,  0.0030],
        [-0.0033, -0.0003,  0.0146,  ..., -0.0075,  0.0043,  0.0084]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.attn.v_proj.lora_b.weight', DTensor(local_tensor=tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.attn.output_proj.weight', DTensor(local_tensor=tensor([[ 0.0247, -0.0133, -0.0262,  ...,  0.0083, -0.0040,  0.0156],
        [ 0.0017, -0.0051, -0.0071,  ...,  0.0052,  0.0020,  0.0098],
        [-0.0088,  0.0178, -0.0083,  ...,  0.0030,  0.0044, -0.0065],
        ...,
        [ 0.0024,  0.0126, -0.0036,  ..., -0.0044, -0.0039,  0.0131],
        [-0.0208,  0.0041,  0.0201,  ..., -0.0012, -0.0129,  0.0112],
        [ 0.0200, -0.0008,  0.0181,  ..., -0.0200,  0.0156, -0.0015]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.mlp.w1.weight', DTensor(local_tensor=tensor([[-0.0114,  0.0176, -0.0256,  ...,  0.0133,  0.0049,  0.0095],
        [-0.0084,  0.0120,  0.0080,  ...,  0.0112,  0.0099,  0.0081],
        [-0.0137,  0.0264, -0.0121,  ...,  0.0067, -0.0087,  0.0024],
        ...,
        [ 0.0036, -0.0014,  0.0247,  ...,  0.0028,  0.0149,  0.0261],
        [-0.0222, -0.0171, -0.0012,  ...,  0.0159, -0.0120, -0.0138],
        [-0.0262, -0.0291, -0.0273,  ...,  0.0369,  0.0400,  0.0075]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.mlp.w2.weight', DTensor(local_tensor=tensor([[-4.0527e-02,  1.8311e-02,  2.0020e-02,  ...,  2.8839e-03,
          1.4771e-02,  3.0212e-03],
        [-5.7983e-03, -2.8076e-03,  2.9907e-03,  ..., -4.6997e-03,
          3.9368e-03, -8.6670e-03],
        [-6.4392e-03, -2.0874e-02, -2.6855e-02,  ...,  5.7678e-03,
         -6.7749e-03, -1.0071e-02],
        ...,
        [ 1.4648e-02,  1.5259e-04, -6.8665e-03,  ..., -1.6602e-02,
         -9.6436e-03,  1.1475e-02],
        [-1.1719e-02,  9.0332e-03,  1.2695e-02,  ..., -1.9775e-02,
          1.4587e-02,  1.1841e-02],
        [-6.6223e-03, -1.5747e-02,  1.3428e-02,  ...,  9.7046e-03,
          5.2691e-05,  3.1891e-03]], device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.mlp.w3.weight', DTensor(local_tensor=tensor([[-0.0178, -0.0300, -0.0054,  ...,  0.0198, -0.0087,  0.0069],
        [ 0.0094,  0.0039,  0.0138,  ...,  0.0079,  0.0405, -0.0069],
        [ 0.0051,  0.0029,  0.0057,  ..., -0.0130,  0.0153, -0.0222],
        ...,
        [ 0.0312,  0.0035,  0.0195,  ...,  0.0054,  0.0123, -0.0003],
        [-0.0027,  0.0182, -0.0096,  ...,  0.0038,  0.0028,  0.0161],
        [ 0.0081, -0.0137, -0.0007,  ..., -0.0048, -0.0034, -0.0116]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.sa_norm.scale', DTensor(local_tensor=tensor([0.4551, 0.3984, 0.4434,  ..., 0.4219, 0.3750, 0.2910], device='cuda:0',
       dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('layers.31._checkpoint_wrapped_module.mlp_norm.scale', DTensor(local_tensor=tensor([0.5117, 0.4785, 0.5078,  ..., 0.5117, 0.4707, 0.4121], device='cuda:0',
       dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('norm.scale', DTensor(local_tensor=tensor([2.4688, 2.3906, 2.5312,  ..., 2.4062, 2.0938, 2.3125], device='cuda:0',
       dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),))), ('output.weight', DTensor(local_tensor=tensor([[ 0.0082,  0.0072,  0.0125,  ...,  0.0095, -0.0215, -0.0138],
        [-0.0070,  0.0041,  0.0190,  ..., -0.0055,  0.0072, -0.0044],
        [ 0.0135,  0.0043,  0.0143,  ...,  0.0006, -0.0087, -0.0150],
        ...,
        [-0.0038,  0.0020,  0.0034,  ...,  0.0003,  0.0081,  0.0083],
        [-0.0038,  0.0020,  0.0034,  ...,  0.0003,  0.0081,  0.0083],
        [-0.0038,  0.0020,  0.0034,  ...,  0.0003,  0.0081,  0.0083]],
       device='cuda:0', dtype=torch.bfloat16), device_mesh=DeviceMesh([0]), placements=(Shard(dim=0),)))]
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