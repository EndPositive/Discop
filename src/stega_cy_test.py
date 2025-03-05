import os
import torch

from config import text_default_settings
from model import get_model, get_tokenizer
from utils import SingleExampleOutput
from stega_cy import encode_text, decode_text

# Load message
message_file_path = os.path.join("temp", "message.txt")
with open(message_file_path, "r", encoding="utf-8") as f:
    message = f.read()
# message *= 10


def test_text():
    settings = text_default_settings
    settings.device = torch.device("cuda:0")

    context = "I remember this film, it was the first film I had watched at the cinema."

    model = get_model(settings)
    tokenizer = get_tokenizer(settings)

    single_example_output: SingleExampleOutput = encode_text(
        model, tokenizer, message, context, settings
    )
    print(single_example_output)
    message_encoded = message[: single_example_output.n_bits]
    message_decoded = decode_text(
        model, tokenizer, single_example_output.stego_object, context, settings
    )
    print(message_encoded)
    print(message_decoded)
    assert message_encoded == message_decoded
