# Hugging Face RoBERTa with Flash Attention 2 :rocket:

Re-implementation of Hugging Face :hugs: [RoBERTa](https://arxiv.org/abs/1907.11692) with [Flash Attention 2](https://tridao.me/publications/flash2/flash2.pdf) in PyTorch.  Drop-in replacement of Pytorch legacy Self-Attention with Flash Attention 2 for Hugging Face RoBERTa based on the standard implementation.

## Installation

You need to install the requirements first, especially `flash-attn` library, which currently is only supported for Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100):

```bash
pip install -r requirements.txt
```

## MLM Demo

```bash
sh demo_mlm.sh
```

You can use any RoBERTa model from Hugging Face model hub and evaluate on any corpus. For example, to use `roberta-base` on `SST-2`:

```bash
python demo_mlm.py --model_class roberta --model_path roberta-base --dataset_name sst2
python demo_mlm.py --model_class flash-roberta --model_path roberta-base --dataset_name sst2
```

## Use with Hugging Face Transformers

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("kiddothe2b/flash-roberta-base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
```