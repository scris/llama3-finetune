### Todo

- [ ] 更新 LoRA 层和 rank，查看结果区别

### Task 1

在单卡 4090 上使用 torchtune 对 Llama-3.1-8B-Instruct 进行了 lora 微调，数据集为 Alpaca Cleaned Dataset（24.1 MB），单 epoch 耗时 2 小时 40 分钟。

在单卡 A6000 上使用 torchtune eleuther_eval 进行基于 truthfulqa_mc2 的模型评估，单次耗时约 3 分钟，最大内存占用 41.92 GB。未微调版本准确率为 0.5404 ± 0.015，微调版本为 0.545 ± 0.0151，差异不大，远低于用 Llama-2-7B-Instruct 微调的结果数据，可能是 Llama-3.1 在训练时已做过较多此方面处理。

### Task 2

继续基于 torchtune 并将场景改换到代码生成，数据集为 Python Code Instructions 18k Alpaca（11.4 MB），在单卡 A6000 上单 epoch 耗时 1 小时 30 分钟。

在单卡 A6000 上使用 torchtune eleuther_eval 进行基于 arc_easy 的模型评估，单次耗时约 1 分钟，最大内存占用 34.68 GB。 未微调版本 acc_norm 为 0.7963 ± 0.0083，微调版本为 0.8035 ± 0.0082，体现了推理能力训练的可迁移性。使用这一数据集来代表的原因是代码生成数据集 code2text_python 执行太慢。同时也基于 truthfulqa_mc2 对微调版本进行完整评估，结果是 0.5186 ± 0.015。

将 layers 28～31 以外的部分的 requires_grad 设为 False，并将这些不更新梯度的层排除在 optimizer 参数之外后，在单卡 A6000 上单 epoch 耗时 50 分钟。这一变更保证仍然可以训练模型的任务特定能力，减少了在通用特征层的计算量。在 arc_easy 的结果是 0.7950 ± 0.0083，在 truthfulqa_mc2 结果是 0.5373 ± 0.015。结果有待提升。

将 layer 25～31 以及 tok_embedding 以外的部分冻结，在单卡 A6000 上单 epoch 耗时 55 分钟。在 arc_easy 的结果是 0.7845 ± 0.0084，在 truthfulqa_mc2 结果是 0.5357 ± 0.0152。看来，需要重新考虑目标为能力迁移的冻结位置。

改换数据集为 Reasoning GSM QnA OA（3.1 MB）以对齐评估数据集，在单卡 4090 上 layers 28～31 单 epoch 耗时 12 分钟。 此时，在 arc_easy 的结果是 0.7992 ± 0.0082，在 truthfulqa_mc2 结果是 0.5367 ± 0.0149。这一结果没有问题。

### Task 3

