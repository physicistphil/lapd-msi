import time
import socket
import struct
import io
import multiprocessing as mp
import numpy as np
import datetime
import tables

from LVBF_buff import LV_fd


def receive_data(q, HOST, PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        while True:
            # Size request
            s.sendall(struct.pack(">i", 1))
            # Wait for response
            data_length = struct.unpack(">i", s.recv(4))[0]

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


class Discharge(tables.IsDescription):
    discharge_current = tables.Float32Col(shape=(4096,))
    discharge_voltage = tables.Float32Col(shape=(4096,))

    interferometer_signals = tables.Float64Col(shape=(1, 4096))
    interferometer_t0_seconds = tables.Int64Col(shape=(1,))
    interferometer_t0_fraction = tables.UInt64Col(shape=(1,))
    interferometer_dt = tables.Float64Col(shape=(1,))

    diode_signals = tables.Float64Col(shape=(3, 4096))
    diode_t0_seconds = tables.Int64Col(3,)
    diode_t0_fraction = tables.Int64Col(3,)
    diode_dt = tables.Float64Col(shape=(3,))

    datarun_timeout = tables.BoolCol()
    datarun_key = tables.StringCol(128)
    datarun_shotnum = tables.UInt32Col()
    datarun_status = tables.UInt16Col()
    datarun_timestamp = tables.Float64Col()

    MSI_timeout = tables.BoolCol()
    MSI_timestamp = tables.Float64Col()

    heater_valid = tables.BoolCol()
    heater_current = tables.Float32Col()
    heater_voltage = tables.Float32Col()
    heater_temp = tables.Float32Col()

    pressure_valid = tables.BoolCol()
    RGA_valid = tables.BoolCol()
    pressure_fill = tables.Float32Col()
    RGA_peak = tables.Float32Col()
    RGA_partials = tables.Float32Col(shape=(50,))

    magnet_valid = tables.BoolCol()
    magnet_profile = tables.Float32Col(shape=(1024,))
    magnet_supplies = tables.Float32Col(shape=(13,))
    magnet_peak = tables.Float32Col()

    internal_shotnum = tables.Int32Col()


def save_data(q, path="saved_MSI/"):
    temp_data = {}
    shotnum = -1
    norun_shotnum = -1
    curr_datarun = ''

    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    norun_h5file = tables.open_file(path + time_str + ".h5", 'a', title="Data--" + time_str)
    norun_group = norun_h5file.create_group("/", 'MSI', 'MSI and diagnostics')
    norun_table = norun_h5file.create_table(norun_group, 'data', Discharge, 'Discharge data')
    norun_shot = norun_table.row

    run_h5file = None

    while True:
        shotnum += 1
        # print("Queue size: {}.".format(q.qsize()), end=' ')
        buffer = q.get()
        reader = LV_fd(endian='>', encoding='cp1252')

        reader.fobj = buffer
        while reader.offset < len(buffer):
            temp_data['discharge_current'] = (reader.read_array(reader.read_numeric, d='>f4'))
            temp_data['discharge_voltage'] = (reader.read_array(reader.read_numeric, d='>f4'))

            temp_data['interferometer_signals'] = (reader.read_array(reader.read_numeric, d='>f8', ndims=2))
            inter_t0 = (reader.read_array(reader.read_timestamp))
            temp_data['interferometer_t0_seconds'] = np.array([inter_t0[i][0] for i in range(inter_t0.shape[0])], dtype=np.int64)
            temp_data['interferometer_t0_fraction'] = np.array([inter_t0[i][1] for i in range(inter_t0.shape[0])], dtype=np.uint64)
            temp_data['interferometer_dt'] = (reader.read_array(reader.read_numeric, d='>f8'))

            temp_data['diode_signals'] = (reader.read_array(reader.read_numeric, d='>f8', ndims=2))
            diode_t0 = (reader.read_array(reader.read_timestamp))
            temp_data['diode_t0_seconds'] = np.array([diode_t0[i][0] for i in range(diode_t0.shape[0])], dtype=np.int64)
            temp_data['diode_t0_fraction'] = np.array([diode_t0[i][1] for i in range(diode_t0.shape[0])], dtype=np.uint64)
            temp_data['diode_dt'] = (reader.read_array(reader.read_numeric, d='>f8'))

            temp_data['datarun_timeout'] = (reader.read_boolean())
            temp_data['datarun_key'] = (reader.read_string())[0]
            temp_data['datarun_shotnum'] = (reader.read_numeric(d='>u4'))
            temp_data['datarun_status'] = (reader.read_numeric(d='>u2'))
            temp_data['datarun_timestamp'] = (reader.read_numeric(d='>f8'))

            temp_data['MSI_timeout'] = (reader.read_boolean())
            temp_data['MSI_timestamp'] = (reader.read_numeric(d='>f8'))

            temp_data['heater_valid'] = (reader.read_boolean())
            temp_data['heater_current'] = (reader.read_numeric(d='>f4'))
            temp_data['heater_voltage'] = (reader.read_numeric(d='>f4'))
            temp_data['heater_temp'] = (reader.read_numeric(d='>f4'))

            temp_data['pressure_valid'] = (reader.read_boolean())
            temp_data['RGA_valid'] = (reader.read_boolean())
            temp_data['pressure_fill'] = (reader.read_numeric(d='>f4'))
            temp_data['RGA_peak'] = (reader.read_numeric(d='>f4'))
            temp_data['RGA_partials'] = (reader.read_array(reader.read_numeric, d='>f4'))

            temp_data['magnet_valid'] = (reader.read_boolean())
            temp_data['magnet_profile'] = (reader.read_array(reader.read_numeric, d='>f4'))
            temp_data['magnet_supplies'] = (reader.read_array(reader.read_numeric, d='>f4'))
            temp_data['magnet_peak'] = (reader.read_numeric(d='>f4'))

            temp_data["internal_shotnum"] = shotnum

        # Switch files if the datarun changed.
        if curr_datarun != temp_data['datarun_key']:
            if run_h5file is not None:
                run_h5file.close()

            curr_datarun = temp_data['datarun_key']
            run_h5file = tables.open_file(path + temp_data['datarun_key'] + ".h5", 'a',
                                          title="Data--" + temp_data['datarun_key'])
            if run_h5file.__contains__("/MSI"):
                run_group = run_h5file.get_node("/MSI")
                run_table = run_h5file.get_node("/MSI/data")
            else:
                run_group = run_h5file.create_group("/", 'MSI', 'MSI and diagnostics')
                run_table = run_h5file.create_table(run_group, 'data', Discharge, 'Discharge data')
            # except tables.exceptions.NodeError:
            #     run_group = run_h5file.MSI
            run_shot = run_table.row

        # Save to separate file if datarun timed out, else save to datarun file.
        if temp_data['datarun_timeout'] == 1:
            norun_shotnum += 1
            for key in temp_data:
                norun_shot[key] = temp_data[key]
            norun_shot.append()
            norun_table.flush()
            print("Saved: norun shot {}. ".format(shotnum), end="")
        else:
            for key in temp_data:
                run_shot[key] = temp_data[key]
            run_shot.append()
            run_table.flush()
            print("Saved: " + temp_data['datarun_key'] + " shot {}. ".format(shotnum), end="")

        if norun_shotnum >= 16383:
            norun_shotnum = -1
            norun_h5file.close()

            # Open new file
            time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
            norun_h5file = tables.open_file(path + time_str + ".h5", 'a', title="Data--" + time_str)
            norun_group = norun_h5file.create_group("/", 'MSI', 'MSI and diagnostics')
            norun_table = norun_h5file.create_table(norun_group, 'data', Discharge, 'Discharge data')
            norun_shot = norun_table.row

        max_diode = np.max(np.array(temp_data['diode_signals'][0]))
        ts_sec = temp_data['diode_t0_seconds'][0]
        ts_frac = temp_data['diode_t0_fraction'][0]
        ts_datarun = datetime.datetime.fromtimestamp(int(temp_data['datarun_timestamp']))
        ts_datarun_frac = np.modf(temp_data['datarun_timestamp'])[0]
        if temp_data['datarun_timeout'] == 1:
            print("Max diode reading: {}. "
                  "MSI time: {}.{}".format(max_diode, ts_sec, ts_frac))
        else:
            print("Shotnum {}. Time: {}.{:0.4}. Max diode reading: {}. "
                  "MSI time: {}.{}".format(temp_data['datarun_shotnum'], ts_datarun,
                                           ts_datarun_frac, max_diode, ts_sec, ts_frac))


if __name__ == '__main__':
    HOST = '192.168.7.54'
    PORT = 27182
    savepath = "saved_MSI/"

    q = mp.Queue()
    # p_read, p_write = mp.Pipe(duplex=False)

    tcp_process = mp.Process(target=receive_data, args=(q, HOST, PORT))
    save_process = mp.Process(target=save_data, args=(q, savepath))

    tcp_process.start()
    save_process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tcp_process.join()
        save_process.join()
