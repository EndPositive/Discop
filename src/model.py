from transformers import GPT2Tokenizer, GPT2LMHeadModel

from transformers import PreTrainedTokenizer, PreTrainedModel
from config import Settings


def get_model(settings: Settings) -> PreTrainedModel:
    if settings.model_name in ['gpt2', 'distilgpt2']:
        model = GPT2LMHeadModel.from_pretrained(settings.model_name).to(settings.device)
    else:
        raise NotImplementedError
    model.eval()
    return model


def get_tokenizer(settings: Settings) -> PreTrainedTokenizer:
    if settings.model_name in ['gpt2', 'distilgpt2']:
        tokenizer = GPT2Tokenizer.from_pretrained(settings.model_name)  # local_files_only=True
    else:
        raise NotImplementedError
    return tokenizer
