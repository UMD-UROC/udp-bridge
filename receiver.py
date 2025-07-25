import cv2
import json
import numpy as np
from reedsolo import RSCodec
import socket
import struct

RX_IP = "127.0.0.1"
RX_PORT = 5005

FRAGMENT_SIZE = 1024
K = 32
N = 40

def main():
    rsc = RSCodec(N - K)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((RX_IP, RX_PORT))

    buffers = {}

    print("Listening")

    while True:
        data, addr = sock.recvfrom(FRAGMENT_SIZE + 8)
        msg_id, frag_idx, total = struct.unpack("<IHH", data[:8])
        fragment = data[8:]

        if msg_id not in buffers:
            buffers[msg_id] = [None] * total
            print(f"Receiving message {msg_id}...")
        
        buffers[msg_id][frag_idx] = fragment

        received = sum(1 for x in buffers[msg_id] if x)
        if received >= K:
            print(f"Received {received}/{total} fragments. Attempting to decode")
            joined = b''.join(frag if frag else b'\x00' * FRAGMENT_SIZE for frag in buffers[msg_id])
            try:
                decoded = rsc.decode(joined)[0]
                image_len = struct.unpack("<I", decoded[:4])[0]
                image_data = decoded[4:4 + image_len]
                feature_data = decoded[4 + image_len:].rstrip(b'\x00')

                image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite("img_dest/received_small.jpg", image)

                features = json.loads(feature_data.decode("utf-8"))
                with open("feature_dest/received_features.json", "w") as f:
                    json.dump(features, f, indent=2)

                print("Message decoded and saved.")
                del buffers[msg_id]

            except Exception as e:
                print(f"Decode failed: {e}")

if __name__ == "__main__":
    main()