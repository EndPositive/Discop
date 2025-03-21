import os
import random

import torch

from settings import text_default_settings

from stega_cy import encode, decode
from subdomain_gpt.loader import load_model

# Load message
message_file_path = os.path.join("temp", "message.txt")
with open(message_file_path, "r", encoding="utf-8") as f:
    message = f.read()

# random start index
message_len = 200 * 8
start_index = random.randint(0, len(message) - message_len)
end_index = start_index + message_len
request_bits = message[start_index:end_index]


def test_text():
    settings = text_default_settings
    settings.device = torch.device("cuda:0")

    model, tokenize, detokenize = load_model(settings.device)

    settings.eos_token_id = tokenize("\n")[0]
    settings.dot_token_id = tokenize(".")[0]
    settings.dash_token_id = tokenize("-")[0]

    context = "."
    context_tokens = torch.tensor(
        tokenize(context), dtype=torch.long, device=settings.device
    )[None, ...]

    message_output = encode(model, context_tokens, request_bits, settings)
    print(message_output)
    assert (
        message_output.n_bits >= message_len
    ), "we should always encode the message (+ padding)"
    query_text = detokenize(message_output.generated_ids)

    print(f"Message text: {query_text}")
    stego = torch.tensor(
        message_output.generated_ids, dtype=torch.long, device=settings.device
    )[None, ...]

    context_and_stego = torch.cat((context_tokens, stego), dim=1)

    response_bits: str = decode(
        model,
        context_and_stego,
        stego_start_index=context_tokens.size(1),
        stego_end_index=context_tokens.size(1) + stego.size(1) - 1,
        settings=settings,
    )

    assert len(response_bits) == len(request_bits)
    assert request_bits == response_bits
