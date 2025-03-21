import os
import random
import torch
import torch.nn.functional as F

from subdomain_gpt import model
from settings import Settings


# Sampling (Encoding) results and statistics for single example
class SingleExampleOutput:
    def __init__(self, generated_ids, n_bits, total_entropy, settings):
        self.generated_ids = generated_ids
        self.temp = settings.temp
        self.n_bits = n_bits
        if generated_ids is not None:
            self.n_tokens = len(generated_ids)
        self.total_entropy = total_entropy
        self.embedding_rate = n_bits / self.n_tokens
        self.utilization_rate = n_bits / total_entropy if total_entropy != 0 else 0

    def __str__(self) -> str:
        d = self.__dict__
        excluded_attr = ['generated_ids']
        selected_attr = list(d.keys())
        for x in excluded_attr:
            selected_attr.remove(x)
        return '\n'.join('{} = {}'.format(key, d[key]) for key in selected_attr)


def set_seed(sd):
    random.seed(sd)


def gen_random_message(seed=None, length: int = 1000, save_path: str = os.path.join('temp', 'message.txt')) -> None:
    # Generating binary message (str) randomly via build-in `random` lib
    import random
    random.seed(seed)

    message = ''
    for _ in range(length):
        message += str(random.randint(0, 1))
    print(message)

    if save_path is None:
        return message
    with open(save_path, 'w', encoding='utf-8') as fout:
        fout.write(message)

@torch.no_grad()
def get_logits(
        model: model.GPT,
        idx: torch.Tensor,
        settings: Settings,
):
    # first, get logits from the model
    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
    logits, _ = model(idx_cond)
    return logits

@torch.no_grad()
def get_probs_indices(
        logits: torch.Tensor,
        settings: Settings,
        forbidden_token_ids: list,
) -> tuple:
    logits = logits[0, -1, :].to(settings.device)
    for forbidden_token_id in forbidden_token_ids:
        logits[forbidden_token_id] = float("-inf")
    logits, indices = logits.sort(descending=True)
    logits = logits.double()
    indices = indices.int()

    if settings.temp is None:
        settings.temp = 1.0
    logits_temp = logits / settings.temp
    probs = F.softmax(logits_temp, dim=-1)

    return probs, indices


if __name__ == '__main__':
    gen_random_message(length=1000000)
