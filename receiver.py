import cv2
import json
import numpy as np
import socket
import struct
import sys
import traceback
from zfec.easyfec import Decoder

RX_IP = "127.0.0.1"
RX_PORT = 5005

FRAGMENT_SIZE = 1024

FRAG_KEY = 'fragments'
K_KEY = 'K'
N_KEY = 'N'
NUM_BLOCK_KEY = 'num_blocks'
IDX_KEY = 'indices'
PADLEN_KEY = 'padlen'
DECODER_KEY = 'decoder'
COMPLETE_BLOCKS_KEY = 'complete_blocks'

def main():
    buffers = {}
    complete_images = []
    
    # open socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # increase receive buffer to 4 MB
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)

    sock.bind((RX_IP, RX_PORT))

    print("Listening")

    while True:
        # pull data from socket, one fragment at a time
        data, _ = sock.recvfrom(FRAGMENT_SIZE + 20)

        # unpack UDP packet
        msg_id, frag_idx, total, k_val, padlen, block_idx, num_blocks = struct.unpack("<IHHHIIH", data[:20])
        fragment = data[20:]

        # check if we have already completed this image, discard extra packets
        if msg_id in complete_images:
            print(f"Already completed image id: {msg_id}, ignoring fragment: {frag_idx}, block {block_idx}")
            continue
        # do some basic error checking based on received packet information.
        # partial validation that the first couple bytes of the packet match the expected header format.
        if k_val >= total:
            print(f"[ERROR] Invalid FEC parameters: K >= N for message: {msg_id}, fragment: {frag_idx}, block: {block_idx}")
            print(f"  Got K: {k_val}, N: {total}")
            continue
        
        # new "message" received. log and start new message dict in memory
        if msg_id not in buffers:
            print(f"MESSAGE: {msg_id}")
            buffers[msg_id] = {
                N_KEY : total,
                K_KEY: k_val,
                NUM_BLOCK_KEY: num_blocks,
                DECODER_KEY: Decoder(k_val, total),
                COMPLETE_BLOCKS_KEY: {}
            }
        # secondary fragment of existing message. validate its header values against assumed good first fragment header values
        else:
            if k_val != buffers[msg_id][K_KEY] or total != buffers[msg_id][N_KEY] or num_blocks != buffers[msg_id][NUM_BLOCK_KEY]:
                print(f"[ERROR] Inconsistent K/N in block: {block_idx}, fragment: {frag_idx} of message {msg_id}. Discarding fragment.")
                print(f"  Expected K/N/NumBlocks {buffers[msg_id][K_KEY]}/{buffers[msg_id][N_KEY]}/{buffers[msg_id][NUM_BLOCK_KEY]}, got {k_val}/{total}/{num_blocks}")
                continue

        # check if fragment is representative of a block that has already been completed
        if block_idx in buffers[msg_id][COMPLETE_BLOCKS_KEY]:
            # print(f"Block {block_idx}, Frag {frag_idx}, ignored (completed block).")
            continue
        
        # check if the fragment is representative of a new block
        if block_idx not in buffers[msg_id]:
            print(f"BLOCK: {block_idx}")
            buffers[msg_id][block_idx] = {
                FRAG_KEY : [None] * total,
                IDX_KEY: [],
                PADLEN_KEY: padlen
            }
        # secondary fragment of existing block. validate its header values against assumed good first fragment header values
        else:
            if padlen != buffers[msg_id][block_idx][PADLEN_KEY]:
                print(f"[ERROR]Inconsistent padlen in block: {block_idx}, fragment {frag_idx} of message {msg_id}. Discarding fragment.")
                print(f"  Expected {buffers[msg_id][block_idx][PADLEN_KEY]}, got {padlen}")
                continue
        
        # check if this fragment has been received before
        if buffers[msg_id][block_idx][FRAG_KEY][frag_idx] is None:
            buffers[msg_id][block_idx][FRAG_KEY][frag_idx] = fragment
            buffers[msg_id][block_idx][IDX_KEY].append(frag_idx)
        else:
            print(f"Block {block_idx}, Frag {frag_idx}, ignored (duplicate frag).")
            continue
        
        frag_count = len(buffers[msg_id][block_idx][IDX_KEY])
        #print(f"Block {block_idx}, Frag {frag_idx}, received ({frag_count}/{total}).")
        
        # check if minimum fragment threshold has been reached for message reconstruction
        if frag_count >= buffers[msg_id][K_KEY]:
            print(f"[BLOCK DECODE] Received {frag_count}/{total} fragments for message {msg_id}, block {block_idx}. Attempting to decode")
            
            indices = buffers[msg_id][block_idx][IDX_KEY]
            fragments = [buffers[msg_id][block_idx][FRAG_KEY][i] for i in indices]

            try:
                # create decoder based on K and N from the first header for this msg_id
                decoder = buffers[msg_id][DECODER_KEY]
                block_payload = decoder.decode(fragments, indices, buffers[msg_id][block_idx][PADLEN_KEY])
                buffers[msg_id][COMPLETE_BLOCKS_KEY][block_idx] = block_payload
                del buffers[msg_id][block_idx]
            except Exception as e:
                print(f"Msg: {msg_id}, Block {block_idx} decode failed: {e}")
                traceback.print_exc()
                del buffers[msg_id]
                sys.exit()

        blocks_received = len(buffers[msg_id][COMPLETE_BLOCKS_KEY])
        blocks_expected = buffers[msg_id][NUM_BLOCK_KEY]
        if blocks_received >= blocks_expected:
            print(f"[MESSAGE DECODE] Received {blocks_received}/{blocks_expected}, attempting file reconstruction")
            full_payload = b''.join(buffers[msg_id][COMPLETE_BLOCKS_KEY][i] for i in sorted(buffers[msg_id][COMPLETE_BLOCKS_KEY]))
            try:
                # pull image length header value from payload
                image_len = struct.unpack("<I", full_payload[:4])[0]
                # pull image bytes from payload
                image_data = full_payload[4:4 + image_len]
                # pull feature data from payload
                feature_data = full_payload[4 + image_len:].rstrip(b'\x00')

                # decode image from bytes to jpg
                image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite("img_dest/received.jpg", image)

                # re-jsonify feature data
                features = json.loads(feature_data.decode("utf-8"))
                with open("feature_dest/received_features.json", "w") as f:
                    json.dump(features, f, indent=2)

                print("Message decoded and saved.")
                del buffers[msg_id]
                complete_images.append(msg_id)

            except Exception as e:
                print(f"Image reconstruction failed for message: {msg_id}: {e}")
                traceback.print_exc()
                del buffers[msg_id]
                sys.exit()

if __name__ == "__main__":
    main()