import socket

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from config import Settings, text_default_settings
from model import get_model, get_tokenizer
from utils import SingleExampleOutput
from stega_cy import encode_text, decode_text

query_header_length = 5


def main():
    HOST = "localhost"
    PORT = 12345

    settings: Settings = text_default_settings
    settings.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context = "I remember this film, it was the first film I had watched at the cinema."

    model: PreTrainedModel = get_model(settings=settings)
    tokenizer: PreTrainedTokenizer = get_tokenizer(settings=settings)

    with socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) as s:
        s.bind((HOST, PORT))
        print(f"UDP server listening on {HOST}:{PORT}")

        while True:
            query_packet, addr = s.recvfrom(4096)

            try:
                if len(query_packet) < query_header_length:
                    raise ValueError(
                        f"UDP message too short: received {len(query_packet)} bytes, expected at least {query_header_length} bytes"
                    )

                query_data_length: int = int.from_bytes(query_packet[0:4], "big")

                if len(query_packet) < query_header_length + query_data_length:
                    raise ValueError(
                        f"UDP message too short: received {len(query_packet)} bytes, expected {query_header_length + query_data_length} bytes"
                    )

                should_decode: int = int.from_bytes(query_packet[4:5], "big")
                query_data: bytes = query_packet[5 : 5 + query_data_length]
                if should_decode:
                    query_text: str = query_data.decode()
                    print(f"Message text: {query_text}")

                    response_bits: str = decode_text(
                        model,
                        tokenizer,
                        query_text,
                        context,
                        settings,
                    )

                    response_data: bytes = int(response_bits, 2).to_bytes(
                        (len(response_bits) + 7) // 8, "big"
                    )
                    s.sendto(response_data, addr)
                else:
                    request_bits: str = "".join(
                        format(byte, "08b") for byte in query_data
                    )
                    print(f"Message bits: {request_bits}")

                    message_output: SingleExampleOutput = encode_text(
                        model, tokenizer, request_bits, context, settings
                    )
                    query_text = message_output.stego_object

                    s.sendto(query_text.encode(), addr)

                print(f"Sent response to {addr}")
            except Exception as e:
                print(f"Error processing message from {addr}: {e}")
                s.sendto(b"", addr)


if __name__ == "__main__":
    main()
