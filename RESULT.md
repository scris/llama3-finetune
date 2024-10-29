### Task 1

在单卡 4090 上使用 torchtune 对 Llama-3.1-8B-Instruct 进行了 lora 微调，数据集为 Alpaca Cleaned Dataset，单 epoch 耗时 2 小时 40 分钟。

在单卡 A6000 上使用 torchtune eleuther_eval 进行基于 truthfulqa_mc2 的模型评估，单次耗时约 3 分钟，最大内存占用 41.92 GB。

未微调版本准确率为 0.5404 ± 0.015，微调版本为 0.545 ± 0.0151，差异不大，远低于用 Llama-2-7B-Instruct 微调的结果数据，可能是 Llama-3.1 在训练时已做过较多此方面处理。

### Task 2

继续基于 torchtune 并将场景改换到代码生成。