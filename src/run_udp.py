import os
import socket

import torch

from settings import Settings, text_default_settings
from stega_cy import encode, decode
from subdomain_gpt.loader import load_model


def main():
    HOST = "127.0.0.1"
    PORT = int(os.getenv("PORT"))

    settings: Settings = text_default_settings
    settings.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, tokenize, detokenize = load_model(settings.device)

    settings.eos_token_id = tokenize("\n")[0]
    settings.dot_token_id = tokenize(".")[0]
    settings.dash_token_id = tokenize("-")[0]

    context = "."
    context_tokens = torch.tensor(
        tokenize(context), dtype=torch.long, device=settings.device
    )[None, ...]

    with socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) as s:
        s.bind((HOST, PORT))
        print(f"UDP server listening on {HOST}:{PORT}")

        while True:
            query_packet, addr = s.recvfrom(4096)

            try:
                if len(query_packet) < 2:
                    raise ValueError(
                        f"UDP message too short: received {len(query_packet)} bytes, expected at least {2} bytes"
                    )

                query_data_length: int = int.from_bytes(query_packet[0:1], "big")

                if len(query_packet) < 2 + query_data_length:
                    raise ValueError(
                        f"UDP message too short: received {len(query_packet)} bytes, expected {2 + query_data_length} bytes"
                    )

                should_decode: int = int.from_bytes(query_packet[1:2], "big")
                query_data: bytes = query_packet[2 : 2 + query_data_length]
                if should_decode:
                    print("Decoding message")
                    print("--" * 20)

                    query_text: str = query_data.decode()
                    print(f"Message text: {query_text}")

                    stego = torch.tensor(
                        tokenize(query_text), dtype=torch.long, device=settings.device
                    )[None, ...]

                    context_and_stego = torch.cat((context_tokens, stego), dim=1)

                    response_bits: str = decode(model, context_and_stego, stego_start_index=context_tokens.size(1),
                                                stego_end_index=context_tokens.size(1) + stego.size(1) - 1,
                                                settings=settings)

                    response_bits = response_bits[::-1]
                    print("Response bits", response_bits)

                    response_data: bytes = int(response_bits, 2).to_bytes(
                        (len(response_bits) + 7) // 8, "big"
                    )

                    s.sendto(response_data, addr)
                else:
                    print("Encoding message")
                    print("--" * 20)

                    request_bits: str = "".join(
                        format(byte, "08b") for byte in query_data
                    )
                    print(f"Message bits: {request_bits}")
                    request_bits = request_bits[::-1]

                    message_output = encode(
                        model, context_tokens, request_bits, settings
                    )
                    print(message_output)
                    query_text = detokenize(message_output.generated_ids)
                    print(query_text)

                    bytes_sent = s.sendto(query_text.encode(), addr)
                    print("bytes_sent", bytes_sent)

                print(f"Sent response to {addr}")
            except Exception as e:
                print(f"Error processing message from {addr}: {e}")
                s.sendto(b"", addr)


if __name__ == "__main__":
    main()
