import cv2
import json
import numpy as np
import os
import socket
import struct
from zfec.easyfec import Decoder

RX_IP = "127.0.0.1"
RX_PORT = 5005

FRAGMENT_SIZE = 1024

FRAG_KEY = 'fragments'
K_KEY = 'K'
N_KEY = 'N'
IDX_KEY = 'indices'
PADLEN_KEY = "padlen"

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((RX_IP, RX_PORT))

    buffers = {}

    print("Listening")

    while True:
        data, addr = sock.recvfrom(FRAGMENT_SIZE + 14)
        msg_id, frag_idx, total, k_val, padlen = struct.unpack("<IHHHI", data[:14])
        fragment = data[14:]

        if k_val >= total:
            print(f"Invalid FEC parameters: K >= N for message: {msg_id}, fragment: {frag_idx}")
            print(f"Got K: {k_val}, N: {total}")
            continue
        
        if msg_id not in buffers:
            print(f"Receiving message {msg_id} K: {k_val}, N: {total}")
            buffers[msg_id] = {
                FRAG_KEY : [None] * total,
                IDX_KEY: [],
                N_KEY : total,
                K_KEY: k_val,
                PADLEN_KEY: padlen
            }
            
        else:
            if k_val != buffers[msg_id][K_KEY] or total != buffers[msg_id][N_KEY]:
                print(f"Inconsistent K/N in fragment {frag_idx} of message {msg_id}. Discarding fragment.")
                print(f"Expected K/N {buffers[msg_id][K_KEY]}/{buffers[msg_id][N_KEY]}, got {k_val}/{total}")
                continue
        
        if buffers[msg_id][FRAG_KEY][frag_idx] is None:
            buffers[msg_id][FRAG_KEY][frag_idx] = fragment
            buffers[msg_id][IDX_KEY].append(frag_idx)
        else:
            print(f"Duplicate fragment {frag_idx} for message {msg_id} ignored.")

        received = len(buffers[msg_id][IDX_KEY])
        print(f"Fragment {frag_idx} received for msg {msg_id} ({received}/{total})")
        
        if received >= buffers[msg_id][K_KEY]:
            print(f"Received {received}/{total} fragments for message {msg_id}. Attempting to decode")
            
            indices = buffers[msg_id][IDX_KEY]
            fragments = [buffers[msg_id][FRAG_KEY][i] for i in indices]

            try:
                decoder = Decoder(buffers[msg_id][K_KEY], buffers[msg_id][N_KEY])
                recovered = decoder.decode(fragments, indices, buffers[msg_id][PADLEN_KEY])
                
                full_payload = recovered
                image_len = struct.unpack("<I", full_payload[:4])[0]
                image_data = full_payload[4:4 + image_len]
                feature_data = full_payload[4 + image_len:].rstrip(b'\x00')

                image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite("img_dest/received_small_plant.jpg", image)

                features = json.loads(feature_data.decode("utf-8"))
                with open("feature_dest/received_features.json", "w") as f:
                    json.dump(features, f, indent=2)

                print("Message decoded and saved.")
                del buffers[msg_id]

            except Exception as e:
                print(f"Decode failed for message {msg_id}: {e}")
                del buffers[msg_id]

if __name__ == "__main__":
    main()