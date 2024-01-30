import time
import socket
import struct
import io
import multiprocessing as mp
import queue
import numpy as np
import datetime
import tables
import mysql.connector
import re
import datetime
import pickle


def receive_data(q, MCAST_GRP, MCAST_PORT):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('', MCAST_PORT))
                # on this port, listen ONLY to MCAST_GRP
                # sock.bind((MCAST_GRP, MCAST_PORT))
                mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

                while True:
                    data_length = struct.unpack(">i", sock.recv(4))[0]
                    data = b''
                    data_remaining = data_length

                    BUFF_LEN = 65535
                    while data_remaining > 0:
                        part = sock.recv(BUFF_LEN if data_remaining > BUFF_LEN else data_remaining)
                        data += part
                        data_remaining -= len(part)

                    if data != b'':
                        q.put(data)

        except Exception as e:
            print("Recieve data exception: ")
            print(repr(e))
            time.sleep(0.5)


def print_msi_info(q):
    while True:
        msi = pickle.loads(q.get())
        temp_data = msi

        ts_sec = datetime.datetime.fromtimestamp(temp_data['diode_t0_seconds'][0])
        ts_frac = temp_data['diode_t0_fraction'][0].astype('u8') / 2 ** 64
        ts_datarun = datetime.datetime.fromtimestamp(int(temp_data['datarun_timestamp']))
        ts_datarun_frac = np.modf(temp_data['datarun_timestamp'])[0]
        if temp_data['datarun_timeout'] == 1:
            print("MSI time: {}+{:0.4}".format(ts_sec, ts_frac), flush=True)
        else:
            print("Datarun shot: {}. Time: {}.{:0.4}. "
                  "MSI time: {}+{:0.4}".format(temp_data['datarun_shotnum'], ts_datarun,
                                           ts_datarun_frac, ts_sec, ts_frac), flush=True)


        # Count number of shots
        # If datarun, save every X amount


if __name__ == '__main__':
    # Multicast group settings
    # '224.0.0.36'
    MCAST_GRP = '224.1.1.1'
    MCAST_PORT = 10004

    q = mp.Queue()

    multicast_process = mp.Process(target=receive_data, args=(q, MCAST_GRP, MCAST_PORT))
    multicast_process.start()

    print_process = mp.Process(target=print_msi_info, args=(q,))
    print_process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupting processes...")
    finally:
        multicast_process.terminate()
        multicast_process.join()
        print_process.terminate()
        print_process.join()
        multicast_process.close()
        print_process.close()
