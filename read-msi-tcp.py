import time
import socket
import struct
import io
import multiprocessing as mp
import numpy as np
import datetime

from LV_binary_files.LVBF_buff import LV_fd


def receive_data(q, HOST, PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        while True:
            # print("Size request")
            s.sendall(struct.pack(">i", 1))
            data_length = struct.unpack(">i", s.recv(4))[0]
            # print("Size: {}".format(data_length))

            data = b''
            data_remaining = data_length

            # print("Data request")
            BUFF_LEN = 1024
            while data_remaining > 0:
                # print(data_remaining)
                s.sendall(struct.pack(">i", 2))
                part = s.recv(BUFF_LEN if data_remaining > BUFF_LEN else data_remaining)
                data += part
                data_remaining -= BUFF_LEN
                # print(data_remaining)
            # print("Data received. Length: {}".format(len(data)))

            q.put(data)


def save_data(q, filename="data.hdf5"):
    class Discharge():
        pass

    shot = Discharge()
    shotnum = -1
    while True:
        shotnum += 1
        # print("Queue size: {}.".format(q.qsize()), end=' ')
        buffer = q.get()
        reader = LV_fd(endian='>', encoding='cp1252')

        reader.fobj = buffer
        while reader.offset < len(buffer):
            shot.discharge_current = (reader.read_array(reader.read_numeric, d='>f4'))
            shot.discharge_voltage = (reader.read_array(reader.read_numeric, d='>f4'))

            shot.interferometer_signals = (reader.read_array(reader.read_numeric, d='>f8', ndims=2))
            shot.interferometer_t0 = (reader.read_array(reader.read_timestamp))
            shot.interferometer_dt = (reader.read_array(reader.read_numeric, d='>f8'))

            shot.diode_signals = (reader.read_array(reader.read_numeric, d='>f8', ndims=2))
            shot.diode_t0 = (reader.read_array(reader.read_timestamp))
            shot.diode_dt = (reader.read_array(reader.read_numeric, d='>f8'))

            shot.datarun_timeout = (reader.read_boolean())
            shot.datarun_key = (reader.read_string())
            shot.datarun_shotnum = (reader.read_numeric(d='>u4'))
            shot.datarun_status = (reader.read_numeric(d='>u2'))
            shot.datarun_timestamp = (reader.read_numeric(d='>f8'))

            shot.MSI_timeout = (reader.read_boolean())
            shot.MSI_timestamp = (reader.read_numeric(d='>f8'))

            shot.heater_valid = (reader.read_boolean())
            shot.heater_current = (reader.read_numeric(d='>f4'))
            shot.heater_voltage = (reader.read_numeric(d='>f4'))
            shot.heater_temp = (reader.read_numeric(d='>f4'))

            shot.pressure_valid = (reader.read_boolean())
            shot.RGA_valid = (reader.read_boolean())
            shot.pressure_fill = (reader.read_numeric(d='>f4'))
            shot.RGA_peak = (reader.read_numeric(d='>f4'))
            shot.RGA_partials = (reader.read_array(reader.read_numeric, d='>f4'))

            shot.magnet_valid = (reader.read_boolean())
            shot.magnet_profile = (reader.read_array(reader.read_numeric, d='>f4'))
            shot.magnet_supplies = (reader.read_array(reader.read_numeric, d='>f4'))
            shot.magnet_peak = (reader.read_numeric(d='>f4'))

        max_diode = np.max(np.array(shot.diode_signals[0]))
        ts_sec = shot.diode_t0['seconds'][0]
        ts_frac = shot.diode_t0['fraction'][0]
        ts_datarun = datetime.datetime.fromtimestamp(int(shot.datarun_timestamp))
        ts_datarun_frac = np.modf(shot.datarun_timestamp)[0]
        if shot.datarun_timeout == 1:
            print("Shotnum (internal) {}. Max diode reading: {}. "
                  "MSI time: {}.{}".format(shotnum, max_diode, ts_sec, ts_frac))
        else:
            print("Shotnum {}. Time: {}.{:0.4}. Max diode reading: {}. "
                  "MSI time: {}.{}".format(shot.datarun_shotnum, ts_datarun, ts_datarun_frac,
                                           max_diode, ts_sec, ts_frac))


if __name__ == '__main__':
    HOST = '192.168.7.54'
    PORT = 27182
    filename = "test.txt"

    q = mp.Queue()
    # p_read, p_write = mp.Pipe(duplex=False)

    tcp_process = mp.Process(target=receive_data, args=(q, HOST, PORT))
    save_process = mp.Process(target=save_data, args=(q, filename))

    tcp_process.start()
    save_process.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        tcp_process.join()
        save_process.join()


# s.makefile
