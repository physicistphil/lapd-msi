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

from LVBF_buff import LV_fd


# Receive data from the MSI system / old housekeeping labview program
def receive_data(q, HOST, PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        # This response-request paradigm is unncessary -- this and the labview code needs to
        #   be edited. It's this way because I didn't quite know what I was doing when I first wrote
        #   it and I was compensating for lost info (but it was because I just didn't check how
        #   long the received data actually was and assumed it was 1024)
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
                data_remaining -= len(part)
                # print(data_remaining)
            # print("Data received. Length: {}".format(len(data)))

            q.put(data)


# Receive data from the spectrometer server
def receive_data_spec(q_spec, HOST, PORT):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))

                while True:
                    # Get data size
                    data_length = struct.unpack(">i", s.recv(4))[0]

                    data = b''
                    data_remaining = data_length

                    BUFF_LEN = 1024
                    while data_remaining > 0:
                        part = s.recv(BUFF_LEN if data_remaining > BUFF_LEN else data_remaining)
                        data += part
                        data_remaining -= len(part)

                    spec_data = np.frombuffer(data, dtype=np.float).reshape(2, -1)
                    q_spec.put(spec_data)
        except Exception as e:
            print("Spectrometer client exception: ")
            print(repr(e))
            time.sleep(0.5)


class Discharge(tables.IsDescription):
    discharge_current = tables.Float32Col(shape=(4096,))
    discharge_voltage = tables.Float32Col(shape=(4096,))

    north_discharge_current = tables.Float32Col(shape=(4096,))
    north_discharge_voltage = tables.Float32Col(shape=(4096,))

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

    spectrometer = tables.Float64Col(shape=(2, 3648))


def save_data(q, q_spec, path="saved_MSI/"):
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

    # Compile the regex pattern for use with the datarun_key
    num_pat = re.compile(r"\b\d+")
    proj_pat = re.compile(r"\w+\.h5")

    # Loop to:
    #   get MSI information
    #   get spectrometer information
    #   switch files (if necessary)
    #   save the info to a h5 file
    while True:
        shotnum += 1
        # print("Queue size: {}.".format(q.qsize()), end=' ')

        # We want this to be a blocking call -- no discharge information = no plasma
        buffer = q.get(block=True, timeout=None)
        reader = LV_fd(endian='>', encoding='cp1252')

        reader.fobj = buffer
        while reader.offset < len(buffer):
            temp_data['discharge_current'] = (reader.read_array(reader.read_numeric, d='>f4'))
            temp_data['discharge_voltage'] = (reader.read_array(reader.read_numeric, d='>f4'))

            temp_data['north_discharge_current'] = (reader.read_array(reader.read_numeric, d='>f4'))
            temp_data['north_discharge_voltage'] = (reader.read_array(reader.read_numeric, d='>f4'))

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

        # Wait at most 0.2 seconds
        try:
            spec_data = q_spec.get(block=True, timeout=0.2)
            # Make sure we get the most recent spectrometer information in case there is a hangup
            # Might want to check if this is actually a problem first.
            # while q_spec.empty() is False:
            #     spec_data = q_spec.get(block=True, timeout=0.2)
            temp_data['spectrometer'] = spec_data
        except queue.Empty:
            print("Spectrometer offline")
            temp_data['spectrometer'] = np.zeros((2, 3648), dtype=np.float)

        # Switch files if the datarun changed.
        if curr_datarun != temp_data['datarun_key']:
            if run_h5file is not None:
                run_h5file.close()

            curr_datarun = temp_data['datarun_key']

            # Try looking up datarun in the MySQL database on the control room PC.
            # This makes it much easier to match up the MSI to the datarun later.
            try:
                num = num_pat.search(curr_datarun + ".h5").__getitem__(0)
                proj = proj_pat.search(curr_datarun + ".h5").__getitem__(0)[:-3]
                cnx = mysql.connector.connect(user='readonly', password='uLTentENEnTIncioUtIO',
                                              host='192.168.7.3')
                cursor = cnx.cursor(buffered=True)
                try:
                    cursor.execute("SELECT data_run_name FROM {}.data_runs WHERE data_run_id in ({});".format(proj, num))
                    (run_name,) = cursor.fetchall()[0]
                except Exception as e:
                    print(e)
                cursor.close()
                cnx.disconnect()
                filename = "{} {} {}".format(proj, num, run_name)
            except Exception as e:
                print(e)
                filename = curr_datarun

            run_h5file = tables.open_file(path + filename + ".h5", 'a',
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
            print("Saved: shot {} in {}. ".format(shotnum, time_str), end="")
        else:
            for key in temp_data:
                run_shot[key] = temp_data[key]
            run_shot.append()
            run_table.flush()
            print("Saved: shot {} in ".format(shotnum) + filename + ". ", end="")

        # This magic number controls how many shots are in one file when there isn't a datarun.
        # 16383 is roughly 1/5th a day. 172800 is roughly two days
        if norun_shotnum >= 86400:
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
            print("Datarun shot: {}. Time: {}.{:0.4}. Max diode reading: {}. "
                  "MSI time: {}.{}".format(temp_data['datarun_shotnum'], ts_datarun,
                                           ts_datarun_frac, max_diode, ts_sec, ts_frac))


if __name__ == '__main__':
    # Old housekeeping computer / connection to get MSI
    HOST = '192.168.7.54'
    PORT = 27182
    # Raspberry pi that has the spectrometer attached (and digitizers eventually)
    HOST_spec = '192.168.7.91'
    PORT_spec = 5004

    savepath = "saved_MSI/"

    q = mp.Queue()
    q_spec = mp.Queue()
    # p_read, p_write = mp.Pipe(duplex=False)

    tcp_process = mp.Process(target=receive_data, args=(q, HOST, PORT))
    tcp_process_spec = mp.Process(target=receive_data_spec, args=(q_spec, HOST_spec, PORT_spec))
    save_process = mp.Process(target=save_data, args=(q, q_spec, savepath))

    tcp_process.start()
    tcp_process_spec.start()
    save_process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupting processes...")
    finally:
        tcp_process.terminate()
        tcp_process.join()
        tcp_process_spec.terminate()
        tcp_process_spec.join()
        save_process.join()
        tcp_process.close()
        tcp_process_spec.close()
        save_process.close()
