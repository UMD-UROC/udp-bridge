import cv2
import json
import math
import numpy as np
import socket
import struct
import sys
import time
from zfec.easyfec import Encoder

RX_IP = "127.0.0.1"
RX_PORT = 5005
MESSAGE_ID = 1

FRAGMENT_SIZE = 1024

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # increase send buffer size to 4MB
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

    # read a local image in
    filename = "large.jpg"
    image = cv2.imread(f"img_src/{filename}")
    if image is None:
        print(f"Error opening image img_src{filename}")
        sys.exit()

    # encode the jpg image into a memory buffer
    _, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    image_bytes = jpeg.tobytes()
    image_len = len(image_bytes)

    # read in local feature file
    feature_file = "starter.json"
    with open(f"feature_src/{feature_file}", "r") as f:
        features = json.dumps(json.load(f)).encode("utf-8")

    # make image length header and create payload
    len_header = struct.pack("<I", image_len)
    payload = len_header + image_bytes + features
    payload_len = len(payload)

    # calculate FEC variables K + N, dependent on payload size
    K, N, num_blocks, block_payload_size = choose_blocks_k_n(payload_len, fragment_size=FRAGMENT_SIZE)
    print(f"Calculated K: {K}, N: {N}, Number of Blocks: {num_blocks}, Block size: {block_payload_size}")
    print(f"For payload of size: {payload_len}")

    if N > 256:
        raise ValueError(f"N={N} exceeds zfec limit (256 fragments max).")
    
    encoder = Encoder(K, N)
    for block_index in range(num_blocks):
        start = block_index * block_payload_size
        end = start + block_payload_size
        block = payload[start: end]
        padded_block = block.ljust(block_payload_size, b'\x00')
        pad_len = block_payload_size - len(block)
        if pad_len != 0:
            print(f"Encoding block {block_index}, with a pad length of: {pad_len}")
        fragments = encoder.encode(padded_block)
        # check for appropriate number of fragments + check individual fragment sizes
        if len(fragments) != N or any(len(frag) != FRAGMENT_SIZE for frag in fragments):
            raise ValueError("FEC encoding did not produce correct fragment sizes.")

        # print(f"Encoded and split into {len(fragments)} fragments of size {FRAGMENT_SIZE}") 
        
        print(f"Sending block: {block_index}")
        for idx, fragment in enumerate(fragments):
            header = struct.pack("<IHHHIIH", MESSAGE_ID, idx, N, K, pad_len, block_index, num_blocks)
            sock.sendto(header + fragment, (RX_IP, RX_PORT))
            # give the receiver time to breathe
            time.sleep(0.001)

    # sleep as a safety buffer before fully closing socket
    time.sleep(0.01)
    sock.close()
    print("done")

def calculate_k_n(payload_len, fragment_size=1024, redundancy_ratio=0.25, max_parity=16):
    K = math.ceil(payload_len / fragment_size)
    parity = min(int(K * redundancy_ratio), max_parity)
    N = K + parity
    return K, N

def choose_blocks_k_n(payload_len, fragment_size=1024, target_redundancy=0.25, max_n=256, min_k=16):
    best = None
    best_efficiency = 0.0

    # Try increasing block sizes (K * fragment_size)
    for K in range(min_k, max_n):
        N = K + int(K * target_redundancy)
        if N > max_n:
            continue

        block_payload_size = K * fragment_size
        num_blocks = math.ceil(payload_len / block_payload_size)

        total_encoded_bytes = num_blocks * N * fragment_size
        total_payload_bytes = payload_len

        efficiency = total_payload_bytes / total_encoded_bytes

        if efficiency > best_efficiency:
            best = {
                "K": K,
                "N": N,
                "num_blocks": num_blocks,
                "block_payload_size": block_payload_size,
                "total_encoded_bytes": total_encoded_bytes,
                "efficiency": efficiency,
            }
            best_efficiency = efficiency
    if best is None:
        raise ValueError("Unable to find K, N for the given payload")

    return best["K"], best["N"], best["num_blocks"], best["block_payload_size"]


if __name__ == "__main__":
    main()