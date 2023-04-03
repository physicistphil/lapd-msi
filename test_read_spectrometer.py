import time
import socket
import struct
import io
import multiprocessing as mp
import numpy as np

HOST = '192.168.7.91'
PORT = 5004

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    while True:
        # Get data size
        data_length = struct.unpack(">i", s.recv(4))[0]

        # print("Data length: {}".format(data_length))

        data = b''
        data_remaining = data_length

        BUFF_LEN = 1024
        while data_remaining > 0:
            part = s.recv(BUFF_LEN if data_remaining > BUFF_LEN else data_remaining)
            data += part
            data_remaining -= len(part)

        spec_data = np.frombuffer(data, dtype=np.float).reshape(2, -1)
        # print("Shape: {}".format(spec_data.shape))
        # print("First 10 from each dim")
        # print(spec_data[0, 0:10])
        # print(spec_data[1, 0:10])
        # print(np.sum(spec_data[1]))
