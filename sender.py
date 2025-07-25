import cv2
import json
import numpy as np
from reedsolo import RSCodec
import socket
import struct
import sys

RX_IP = "127.0.0.1"
RX_PORT = 5005
MESSAGE_ID = 1

FRAGMENT_SIZE = 1024
K = 32
N = 40

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

print("Payload size: ", len(payload))
print("Max allowed size:", K * FRAGMENT_SIZE)

total_len = FRAGMENT_SIZE * K
padded = payload.ljust(total_len, b'\x00')

rsc = RSCodec(N - K)
encoded = rsc.encode(padded)

fragments = [
    encoded[i * FRAGMENT_SIZE:(i + 1) * FRAGMENT_SIZE]
    for i in range(N)
]

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

for idx, fragment in enumerate(fragments):
    header = struct.pack("<IHH", MESSAGE_ID, idx, N)
    sock.sendto(header + fragment, (RX_IP, RX_PORT))

print("done")