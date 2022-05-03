from specDataClass import SpecInfo

import time
import socket
import struct
import io
import multiprocessing as mp
import numpy as np


HOST = '192.168.7.91'
PORT = 5004

trig_val = 0
int_time_micros = 20000

spectrometer = SpecInfo()
spectrometer.setup_spec(trig_val, int_time_micros)

while True:
    try:
        print("Starting server")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((HOST, PORT))
            sock.listen()
            connection, address = sock.accept()
            with connection:
                while True:
                    spec_data = spectrometer.get_spec()
                    spec_shape = spectrometer.get_shape()

                    spec_size_bytes = spec_data.size * spec_data.itemsize
                    connection.send(struct.pack(">i", spec_size_bytes))
                    connection.sendall(spec_data.tobytes())
                    print("-", end="", flush=True)
                    time.sleep(2)
    except Exception as e:
        print("Server raised exception: ")
        print(repr(e))
    time.sleep(0.25)

