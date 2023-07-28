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
```
Running inference...: 100%|██████████| 4329/4329 [00:53<00:00, 81.35it/s]
Model: RobertaForMaskedLM
Time taken for RobertaForMaskedLM: 53.21654486656189
Average inference time for RobertaForMaskedLM: 0.011
Recall@5 for RobertaForMaskedLM: 0.7251097251097252
----------------------------------------
Running inference...: 100%|██████████| 4329/4329 [00:45<00:00, 95.31it/s] 
Model: FlashRobertaForMaskedLM
Time taken for FlashRobertaForMaskedLM: 45.422065019607544
Average inference time for FlashRobertaForMaskedLM: 0.009
Recall@5 for FlashRobertaForMaskedLM: 0.7253407253407254
```

You can use any RoBERTa model from Hugging Face model hub and evaluate on any corpus. For example, to use `roberta-base` on `SST-2`:

```bash
python demo_mlm.py --model_class roberta --model_path roberta-base --dataset_name sst2
python demo_mlm.py --model_class flash-roberta --model_path roberta-base --dataset_name sst2
```

## Use with Hugging Face Transformers

You can also use the standard Hugging Face language modeling `run_mlm.py` script:

```
sh demo_mlm.hf.sh
```

```
RoBERTaForMaskedLM
***** eval metrics *****
  eval_accuracy           =     0.5758
  eval_loss               =     3.0864
  eval_runtime            = 0:00:11.01
  eval_samples            =      10000
  eval_samples_per_second =    908.235
  eval_steps_per_second   =     28.428
  perplexity              =     21.899
Elapsed time: 39.611320195 seconds

FlashRoBERTaForMaskedLM
***** eval metrics *****
  eval_accuracy           =     0.5714
  eval_loss               =     3.0992
  eval_runtime            = 0:00:08.45
  eval_samples            =      10000
  eval_samples_per_second =   1183.238
  eval_steps_per_second   =     37.035
  perplexity              =    22.1808
```

You can also use FlashRoBERTa directly from Hugging Face Transformers:


```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("kiddothe2b/flash-roberta-base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
```


## Citation 

```
@misc{flashroberta,
  title={Hugging Face RoBERTa with Flash Attention 2,
  author={Chalkidis, Ilias},
  year={2023},
  howpublished={Hugging Face Hub}
}

@article{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning,
  author={Dao, Tri},
  year={2023}
}
```