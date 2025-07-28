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
    # read a local image in
    filename = "small_plant.jpg"
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
    K, N = calculate_k_n(payload_len, fragment_size=FRAGMENT_SIZE)
    payload_limit = K * FRAGMENT_SIZE
    print(f"Calculated K: {K}, N: {N}")
    print("Payload size: ", payload_len)
    print(f"Alotted Fragment Space: {payload_limit}")

    # check for issues with payload size/fragment limits
    if payload_len > payload_limit:
        raise ValueError("The payload is larger than the allotted data fragments.")

    if N > 256:
        raise ValueError(f"N={N} exceeds zfec limit (256 fragments max). Use chunking.")

    # pad the payload with extra space for encoding
    padded_payload = payload.ljust(payload_limit, b'\x00')
    pad_len = payload_limit - payload_len

    # Create the encoder
    encoder = Encoder(K, N)
    fragments = encoder.encode(padded_payload)

    # check for appropriate number of fragments + check individual fragment sizes
    if len(fragments) != N or any(len(frag) != FRAGMENT_SIZE for frag in fragments):
        raise ValueError("FEC encoding did not produce correct fragment sizes.")
    
    print(f"Encoded and split into {len(fragments)} fragments of size {FRAGMENT_SIZE}")

    # start python socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # send UDP packets
    for idx, fragment in enumerate(fragments):
        header = struct.pack("<IHHHI", MESSAGE_ID, idx, N, K, pad_len)
        sock.sendto(header + fragment, (RX_IP, RX_PORT))

    # sleep as a safety buffer before fully closing socket
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