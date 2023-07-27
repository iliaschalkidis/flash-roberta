from transformers import RobertaForMaskedLM, AutoTokenizer
from modeling_flashroberta import FlashRobertaForMaskedLM
import torch
from torch.cuda.amp import autocast
import argparse
from datasets import load_dataset
import time
import re
import random


def demo_mlm(corpus,
             model_class: str = 'roberta',
             model_path: str = 'roberta-base',
             n_samples: int = 1000,
             log_predictions: bool = False):

    # Check if you have a GPU with CUDA support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    if model_class == 'roberta':
        model = RobertaForMaskedLM.from_pretrained(model_path).to(device)
    elif model_class == 'flashroberta':
        model = FlashRobertaForMaskedLM.from_pretrained(model_path).to(device)

    # Set the model to evaluation mode (important for inference)
    model.eval()

    current_time = time.time()
    correct_predictions = 0
    for input_doc, mask_token_id in corpus:
        # Generate the masked language model output
        with autocast():
            mask_token_index = torch.where(input_doc == mask_token_id)[1]
            token_logits = model(input_doc)[0]
            mask_token_logits = token_logits[0, mask_token_index, :]
            top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            correct_predictions += int(document['masked_word'] in tokenizer.decode(top_5_tokens))

            if log_predictions:
                print(f'Top 5 tokens for model: {model.__class__.__name__}')
                print('-' * 50)
                for top_k, token in enumerate(top_5_tokens):
                    print('Top {} token: {}'.format(top_k, tokenizer.decode([token])))
                print('-' * 50)

    end_time = time.time() - current_time
    print(f'Model: {model.__class__.__name__}')
    print(f'Time taken for {model.__class__.__name__}: {end_time}')
    print(f'Recall@5 for {model.__class__.__name__}: {correct_predictions / n_samples}')


def mask_random_words(document):
    # Tokenize the text into words using regular expression
    words = re.findall(r'\b\w+\b', document['text'])

    # Randomly select a word to mask
    word_to_mask = words[random.choice(range(min(200, len(words))))]

    # Replace the selected word with the mask token
    document['text'] = document['text'].replace(word_to_mask, "<mask>", 1)
    document['masked_word'] = word_to_mask

    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with argparse")
    parser.add_argument("--model_class", type=str, default="roberta", help="Model class to use")
    parser.add_argument("--model_path", type=str, default="roberta-base", help="Model path to use")
    parser.add_argument("--dataset_name", type=str, default="c4", help="Dataset to use")
    parser.add_argument("--dataset_config", type=str, default="en", help="Dataset config to use")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to use")

    args = parser.parse_args()

    # Load C4 dataset, take 1000 samples, and mask random words
    c4_dataset = load_dataset(args.dataset_name, args.dataset_config, split="train", streaming=True)
    c4_subset = c4_dataset.take(args.n_samples)
    c4_subset = c4_subset.map(mask_random_words)

    # Pre-encode C4 dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    mask_token_id = torch.tensor(tokenizer.mask_token_id).to(device)
    precompiled_c4_subset = []
    for document in c4_subset:
        input_doc = tokenizer.encode(document['text'], return_tensors="pt", truncation=True).to(device)
        precompiled_c4_subset.append((input_doc, mask_token_id))

    demo_mlm(precompiled_c4_subset, args.model_class, args.model_path, args.n_samples)
