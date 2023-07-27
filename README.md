# HuggingFace RoBERTa with Flash Attention 2 :rocket:

Re-implementation of Hugging Face :hugs: [RoBERTa](https://arxiv.org/abs/1907.11692) with [Flash Attention 2](https://tridao.me/publications/flash2/flash2.pdf) in PyTorch.  Drop-in replacement of Pytorch legacy Self-Attention with Flash Attention 2 for Hugging Face RoBERTa based on the standard implementation.

## Installation and Use

```bash
pip install -r requirements.txt
sh demo_mlm.sh
```

You can use any RoBERTa model from Hugging Face model hub and evaluate on any corpus. For example, to use `roberta-base` on `SST-2`:

```bash
python demo_mlm.py --model_class roberta --model_path roberta-base --dataset_name sst2
python demo_mlm.py --model_class flash-roberta --model_path roberta-base --dataset_name sst2

```