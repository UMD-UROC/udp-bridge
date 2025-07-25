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
    filename = "small_plant.jpg"
    image = cv2.imread(f"img_src/{filename}")
    if image is None:
        print(f"Error opening image img_src{filename}")
        sys.exit()

    _, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    image_bytes = jpeg.tobytes()

    feature_file = "starter.json"
    with open(f"feature_src/{feature_file}", "r") as f:
        features = json.dumps(json.load(f)).encode("utf-8")

    image_len = struct.pack("<I", len(image_bytes))
    payload = image_len + image_bytes + features

    K, N = calculate_k_n(len(payload), fragment_size=FRAGMENT_SIZE)
    print(f"Calculated K: {K}, N: {N}")
    print("Payload size: ", len(payload))
    print(f"Alotted Fragment Space: {K * FRAGMENT_SIZE}")

    if len(payload) > (K * FRAGMENT_SIZE):
        raise ValueError("The payload is larger than the allotted data fragments.")

    if N > 256:
        raise ValueError(f"N={N} exceeds zfec limit (256 fragments max). Use chunking.")

    total_len = FRAGMENT_SIZE * K
    padded = payload.ljust(total_len, b'\x00')
    padlen = total_len - len(payload)

    encoder = Encoder(K, N)
    fragments = encoder.encode(padded)

    if len(fragments) != N or any(len(frag) != FRAGMENT_SIZE for frag in fragments):
        raise ValueError("FEC encoding did not produce correct fragment sizes.")
    
    print(f"Encoded and split into {len(fragments)} fragments of size {FRAGMENT_SIZE}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    for idx, fragment in enumerate(fragments):
        header = struct.pack("<IHHHI", MESSAGE_ID, idx, N, K, padlen)
        sock.sendto(header + fragment, (RX_IP, RX_PORT))

    time.sleep(0.01)
    sock.close()
    print("done")

def calculate_k_n(payload_len, fragment_size=1024, redundancy_ratio=0.25, max_parity=16):
    K = math.ceil(payload_len / fragment_size)
    parity = min(int(K * redundancy_ratio), max_parity)
    N = K + parity
    return K, N

if __name__ == "__main__":
    main()