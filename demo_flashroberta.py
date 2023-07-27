from transformers import RobertaForMaskedLM, AutoTokenizer
from modeling_flashroberta import FlashRobertaForMaskedLM
import torch
from torch.cuda.amp import autocast
import argparse


def demo_mlm(model_class: str = 'roberta'):

    # Check if you have a GPU with CUDA support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    if model_class == 'roberta':
        model = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
    elif model_class == 'flashroberta':
        model = FlashRobertaForMaskedLM.from_pretrained('roberta-base').to(device)

    # Prepare the input sequence
    sequence = f"My favourite pet is a {tokenizer.mask_token}."
    print('Example sequence:', sequence)
    input = tokenizer.encode(sequence, return_tensors="pt").to(device)

    # Ensure mask_token_id is of type torch.float16
    mask_token_id = torch.tensor(tokenizer.mask_token_id).to(device)

    # Generate the masked language model output
    with autocast():
        mask_token_index = torch.where(input == mask_token_id)[1]

        token_logits = model(input)[0]

        mask_token_logits = token_logits[0, mask_token_index, :]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

        print(f'Top 5 tokens for model: {model.__class__.__name__}')
        print('-' * 50)
        for top_k, token in enumerate(top_5_tokens):
            print('Top {} token: {}'.format(top_k, tokenizer.decode([token])))
        print('-' * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with argparse")
    parser.add_argument("--model_class", type=str, default="roberta", help="Model class to use")

    args = parser.parse_args()

    demo_mlm(args.model_class)
