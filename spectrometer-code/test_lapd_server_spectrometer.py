from specDataClass import SpecInfo

import time
import socket
import struct
import io
import multiprocessing as mp
import numpy as np
import errno
import queue
import select


# Send data to a connection
def send_data(connection, address, queue):
    # with connection:
    alive = True
    while alive:
        try:
            print("=== getting from queue ===")
            trigger_val, integration_time, spec_data, spec_shape = queue.get()
            print("=== read from queue ===")

            spec_size_bytes = spec_data.size * spec_data.itemsize

            print("=== sending info ===")
            connection.send(struct.pack(">i", trigger_val))
            connection.send(struct.pack(">i", integration_time))
            connection.send(struct.pack(">i", spec_size_bytes))
            print("=== sending data ===")
            connection.sendall(spec_data.tobytes())
            print(">", end="", flush=True)
        except socket.error as e:
            if e.errno == errno.EPIPE:
                print("Error/client disconnected")
                # connection.close()
                alive = False
        except KeyboardInterrupt:
            # print(e)
            print("Closing data connection (keyboard interrupt)")
            # connection.close()
            alive = False
        except Exception as e:
            print(e)
            print("Closing data connection")
            # connection.close()
            alive = False
    # connection.close()


# Process grab spectrometer settings from some client and then sends it to the queue
def get_spec_settings(HOST, PORT, config_queue):
    print("Starting config server")
    sock_config = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_config.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_config.bind((HOST, PORT))
    sock_config.listen(5)

    while True:
        try:
            connection, address = sock_config.accept()
            print("New config connection: {}".format(address), flush=True)

            while True:
                # recieve data
                trig_val = struct.unpack(">i", connection.recv(4))[0]
                int_time_micros = struct.unpack(">i", connection.recv(4))[0]
                try:
                    config_queue.put_nowait((trig_val, int_time_micros))
                except queue.Full:
                    print("Config queue full")
                    config_queue.get()
                    config_queue.put_nowait((trig_val, int_time_micros))
        except KeyboardInterrupt:
            # print(e)
            print("Closing config connection (keyboard interrupt)")
            try:
                connection.close()
            except Exception:
                print("Config: no connection to close")
            finally:
                return
        except Exception as e:
            print("Config server raised exception: ", end='')
            print(repr(e))


# Using a data queue list because using one queue for everything is a bit of a mess (concurrency
#   issues) and using shared memory would also be a mess. Hopefully there's a better way.
# This sets up the spectrometer and updates the settings from the config queue. The data is then
#   collected and added to the send-data queues.
def get_spec_data_and_config(config_queue, data_queue_list):
    # check if config queue has something, and if so reconfigure spectrometer
    # collect data and add to queue
    print("Spectrometer thread launched")
    trig_val = 3
    int_time_micros = 20000
    spectrometer = SpecInfo()
    print("Setting up spectrometer...", flush=True)
    spectrometer.open()
    spectrometer.setup_spec(trig_val, int_time_micros)
    spectrometer.setup_spec(trig_val, int_time_micros)

    print("Spectrometer setup completed")

    try:
        while True:
            if config_queue.empty() is False:
                trig_val, int_time_micros = config_queue.get()
                spectrometer.setup_spec(trig_val, int_time_micros)

            spec_data = spectrometer.get_spec()
            spec_shape = spectrometer.get_shape()
            print("-", end="", flush=True)

            if len(data_queue_list) != 0:
                for q in data_queue_list:
                    print("=== putting data in queue ===")
                    try:
                        q.put((trig_val, int_time_micros, spec_data, spec_shape), timeout=0)
                        print("=== putting data in queue success ===")
                    except queue.Full:
                        print("A data queue is full")
                    except Exception as e:
                        print(e)
    except Exception as e:
        print(e)
        print("Closing spectrometer")
        spectrometer.close()
        return


if __name__ == '__main__':
    print("Starting server")

    # Start data manager
    manager = mp.Manager()
    # manager.start()

    connection_process_list = []
    data_queue_list = manager.list()

    HOST = '192.168.7.94'
    PORT = 5004
    PORT_config = 5005

    config_queue = mp.Queue(maxsize=1)
    spec_process = mp.Process(target=get_spec_data_and_config,
                              args=(config_queue, data_queue_list))
    spec_process.start()

    # Start process for getting the spectrometer configuration
    config_process = mp.Process(target=get_spec_settings, args=(HOST, PORT_config, config_queue))
    config_process.start()

    # Main server settings and loop
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((HOST, PORT))
    sock.listen(5)

    try:
        while True:
            try:
                # Wait for somethign to be readable on the socket. Timeout of 0.1
                readable, writeable, exceptional = select.select([sock], [], [sock], 0.1)
                if sock in readable:
                    connection, address = sock.accept()
                    print("New connection: {}".format(address), flush=True)
                    new_queue = manager.Queue(maxsize=1)  # Max size of 1 so only most recent data is sent
                    data_queue_list.append(new_queue)
                    new_process = mp.Process(target=send_data,
                                             args=(connection, address, new_queue))
                    new_process.start()
                    connection_process_list.append(new_process)

                # After making a new connection try to get rid of the old ones
                for p in connection_process_list:
                    idx = connection_process_list.index(p)
                    p.join(timeout=0)
                    if p.is_alive() is False:
                        print("closing process")
                        p.close()
                        connection_process_list.remove(p)
                        del data_queue_list[idx]
            except Exception as e:
                print("Server raised exception: ")
                print(repr(e))

    except KeyboardInterrupt:
        print("Interrupting processes...")
    finally:
        print("Cleaning up")
        sock.close()
        spec_process.terminate()
        spec_process.join()
        spec_process.close()
        config_process.terminate()
        config_process.join()
        config_process.close()

        for proc in connection_process_list:
            proc.terminate()
            proc.join()
            proc.close()
        manager.shutdown()
