# HuggingFace RoBERTa with Flash Attention 2 :rocket:

Re-implementation of Hugging Face :hugs: [RoBERTa](https://arxiv.org/abs/1907.11692) with [Flash Attention 2](https://tridao.me/publications/flash2/flash2.pdf) in PyTorch.  Drop-in replacement of Pytorch legacy Self-Attention with Flash Attention 2 for Hugging Face RoBERTa based on the standard implementation.

## Installation and Use

```bash
pip install -r requirements.txt
sh test.sh
```