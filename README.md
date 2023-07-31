# Hugging Face RoBERTa with Flash Attention 2 :rocket:

Re-implementation of Hugging Face :hugs: [RoBERTa](https://arxiv.org/abs/1907.11692) with [Flash Attention 2](https://tridao.me/publications/flash2/flash2.pdf) in PyTorch.  Drop-in replacement of Pytorch legacy Self-Attention with Flash Attention 2 for Hugging Face RoBERTa based on the standard implementation.
FlashRoBERTa seems to be 20-30% faster compared to the vanilla RoBERTa across all benchmarks (training, inference), without any improvement in memory footprint.

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

You can also use the standard Hugging Face language modeling `run_mlm.py` script for evaluation (inference):

```
sh demo_mlm.hf.sh
```

```
RoBERTaForMaskedLM
---------------------------------------
Eval time: 395.08065152168274
Peak memory: 14.356424704GB
---------------------------------------
***** eval metrics *****
  eval_accuracy           =     0.6667
  eval_loss               =     2.0625
  eval_runtime            = 0:06:30.23
  eval_samples            =     100000
  eval_samples_per_second =    256.257
  eval_steps_per_second   =      4.005
  perplexity              =     7.8655



FlashRoBERTaForMaskedLM
---------------------------------------
Eval time: 296.4356851577759
Peak memory: 14.35724544GB
---------------------------------------
***** eval metrics *****
  eval_accuracy           =     0.6629
  eval_loss               =     2.0794
  eval_runtime            = 0:04:51.75
  eval_samples            =     100000
  eval_samples_per_second =    342.749
  eval_steps_per_second   =      5.357
  perplexity              =        8.0
```

Or training on any corpus:

```
RoBERTaForMaskedLM
---------------------------------------
{'eval_loss': 1.9818187952041626, 'eval_accuracy': 0.6163850658704041, 'eval_runtime': 7.7981, 'eval_samples_per_second': 1282.367, 'eval_steps_per_second': 40.138, 'epoch': 1.0}
{'train_runtime': 669.4028, 'train_samples_per_second': 478.038, 'train_steps_per_second': 14.939, 'train_loss': 2.2284346923828124, 'epoch': 1.0}
Eval time: 669.413816690445
Peak memory: 4.309664768GB

FlashRoBERTaForMaskedLM
---------------------------------------
{'eval_loss': 2.0119481086730957, 'eval_accuracy': 0.6122425376665159, 'eval_runtime': 5.8766, 'eval_samples_per_second': 1701.662, 'eval_steps_per_second': 53.262, 'epoch': 1.0}
{'train_runtime': 551.3029, 'train_samples_per_second': 580.443, 'train_steps_per_second': 18.139, 'train_loss': 2.229952978515625, 'epoch': 1.0}
Eval time: 551.3131585121155
Peak memory: 4.16851456GB
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