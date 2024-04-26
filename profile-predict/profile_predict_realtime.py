import time
import socket
import struct
import multiprocessing as mp
import numpy as np
import datetime
# import tables
import datetime
import pickle
from dateutil.relativedelta import relativedelta

# For GUI
import tkinter as tk
from tkinter.constants import BOTTOM, CENTER, LEFT, TOP, RIGHT, HORIZONTAL
from tkinter.ttk import Progressbar

# For plotting in GUI
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# uhh Other stuff
import gc
import queue

import profile_predict_model
import torch
from collections import OrderedDict
import re


def receive_data(q, MCAST_GRP, MCAST_PORT):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # on this port, listen ONLY to MCAST_GRP
                # sock.bind((MCAST_GRP, MCAST_PORT))
                # mreq = struct.pack("4sl", , socket.INADDR_ANY)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                                socket.inet_aton(MCAST_GRP) + socket.inet_aton('192.168.7.3'))

                sock.bind(('', MCAST_PORT))

                while True:
                    data_length = struct.unpack(">i", sock.recv(4))[0]
                    data = b''
                    data_remaining = data_length
                    # print("Getting data length: " + str(data_length))

                    BUFF_LEN = 65535
                    while data_remaining > 0 and data_length < 1000000:
                        part = sock.recv(BUFF_LEN if data_remaining > BUFF_LEN else data_remaining)
                        data += part
                        data_remaining -= len(part)

                    # print("Received data length: " + str(len(data)))

                    if data_length != len(data):
                        raise Exception("Data length does not match data received")

                    if data != b'':
                        q.put(data)

        except Exception as e:
            print("Recieve data exception: ")
            print(repr(e))
            time.sleep(0.5)


def load_model(checkpoint):
    model = profile_predict_model.ModelClass().to('cpu')
    ckpt = torch.load(checkpoint, map_location='cpu')

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    state_dict = ckpt['model_state_dict']
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    model.load_state_dict(model_dict, strict=True)

    return model


# def infer_profiles(model, data):
#     return model(data)


# Simple boxcar downsampling of time series data
def downsample(arr):
    return np.mean(arr[262:1272].reshape(-1, 101, 10), axis=2).astype('f4')


# Numpy data type for the processed MSI and positions
ds_dtype = [('discharge_current', 'f4', (101,)),
            ('discharge_voltage', 'f4', (101,)),
            # ('interferometer', 'f4', (101,)),
            ('diode_0', 'f4', (101,)),
            ('diode_1', 'f4', (101,)),
            ('diode_2', 'f4', (101,)),
            ('diode_3', 'f4', (101,)),
            ('diode_4', 'f4', (101,)),
            ('magnet_profile', 'f4', (64,)),
            ('pressures', 'f4', (51,)),
            # ('spectrometer', 'f4', (2, ?))
            ('positions', 'f4', (5,))  # receptacle, xyz, theta
            ]


# Takes a single shot of msi and an array of positions to make a profile (axial and radial)
# This single shot needs to have a leading index of 1
def process_msi(msi, sample_positions):
    input_length = sample_positions.shape[0]
    data = np.zeros((input_length,), dtype=ds_dtype)

    data['discharge_current'][:] += downsample(msi['discharge_current'])
    data['discharge_voltage'][:] += downsample(msi['discharge_voltage'])
    diodes = msi['diode_signals']
    data['diode_0'][:] += downsample(diodes[0])
    data['diode_1'][:] += downsample(diodes[1])
    data['diode_2'][:] += downsample(diodes[2])
    diode3 = msi['north_discharge_current']
    diode4 = msi['north_discharge_voltage']
    data['diode_3'][:] += downsample(diode3)
    data['diode_4'][:] += downsample(diode4)
    data['magnet_profile'][:] += np.mean(msi['magnet_profile'].reshape(-1, 64, 16,), axis=2).astype('f4')

    pressure_fill = msi['pressure_fill']
    pressure_RGA = msi['RGA_partials']
    data['pressures'][:] += np.log10(np.concatenate([pressure_fill[np.newaxis].astype('f4'),
                                                     pressure_RGA.astype('f4')], axis=0) + 1e-9)
    data['pressures'] = np.nan_to_num(data['pressures'], nan=-9)
    data['positions'] = sample_positions

    return data


def concat_normalize_inputs(data_in, max_values):
    data = data_in
    num_samples = data.shape[0]
    time_signals = np.concatenate([data['diode_0'][:, :, np.newaxis] / max_values['diode_0'],
                                   data['diode_1'][:, :, np.newaxis] / max_values['diode_1'],
                                   data['diode_2'][:, :, np.newaxis] / max_values['diode_2'],
                                   data['diode_3'][:, :, np.newaxis] / max_values['diode_3'],
                                   data['diode_4'][:, :, np.newaxis] / max_values['diode_4'],
                                   data['discharge_current'][:, :, np.newaxis] / max_values['discharge_current'],
                                   data['discharge_voltage'][:, :, np.newaxis] / max_values['discharge_voltage'],
                                   ], axis=2)

    # pressures = np.nan_to_num(msi['pressures'], nan=-9)
    pressures = data['pressures']
    positions = data['positions']
    # isat = data['isat']
    non_time_signals = np.concatenate([data['magnet_profile'] / max_values['magnet_profile'],
                                       pressures / max_values['pressures'],
                                       positions[:] / max_values['positions']], axis=1)[:, :]
    # Flatten so we can concat non_time_signals on the back
    time_signals = time_signals.reshape(num_samples, -1)

    all_inputs = np.concatenate([time_signals, non_time_signals], axis=1)
    return all_inputs


class App(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root)
        self.root = root
        self.pack()

        # Multicast group settings
        # '224.0.0.36'
        MCAST_GRP = '224.1.1.1'
        MCAST_PORT = 10004

        self.data_q = mp.Queue()
        self.multicast_process = mp.Process(target=receive_data, args=(self.data_q, MCAST_GRP, MCAST_PORT))
        self.multicast_process.start()

        self.create_plot()
        self.update_plot()
        self.create_button("Quit", self._quit, location=BOTTOM, pady=20)
        self.create_button("Save trace", self.save_trace, location=TOP, pady=20)
        self.create_button("Clear traces", self.clear_trace, location=TOP, pady=20)

        self.data_list = []

        # Make position arrays
        recep_1 = np.ones((47, 1), dtype='f4')
        recep_4 = np.ones((47, 1), dtype='f4') * 1
        self.x_loc = np.arange(-22.5, 46.6, 1.5, dtype='f4')[:, np.newaxis]
        y_loc = np.zeros((47, 1), dtype='f4')
        theta_loc = np.zeros((47, 1), dtype='f4')
        z_loc_1 = np.ones((47, 1), dtype='f4') * 894.6
        z_loc_4 = np.ones((47, 1), dtype='f4') * 830.7
        self.pos_recep_1 = np.concatenate([recep_1, self.x_loc, y_loc, z_loc_1, theta_loc], axis=1)
        self.pos_recep_4 = np.concatenate([recep_4, self.x_loc, y_loc, z_loc_4, theta_loc], axis=1)

        # Load max values
        self.max_values = np.load('no-ifo_2024-04-19_21h-29m-27s_model-999-403_max_values.npz')

        # Initialize the ML model
        self.chkpt_path = 'no_ifo_2024-04-19_21h-29m-27s_model-999-403.pt'
        self.model = load_model(self.chkpt_path)

    def _quit(self):
        self.multicast_process.terminate()
        self.multicast_process.join()
        self.root.quit()     # stops mainloop
        self.root.destroy()

    def create_plot(self):
        temp = tk.LabelFrame(self.root)

        self.fig = plt.Figure()

        self.ax = self.fig.add_subplot(211)
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel(r"Isat trace (cm$^{-3}$)")
        self.ax.set_xlim(-1, 31.4)
        self.ax.set_ylim(-3e11, 1.3e13)
        self.ax.set_title("Predicted profiles (T = ?)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=temp)

        self.lines_R1, = self.ax.plot(np.zeros(76), np.zeros(76), label='Port 25 (z = 894.6 cm)\nx = 0')
        self.lines_R4, = self.ax.plot(np.zeros(76), np.zeros(76), label='Port 27 (z = 830.7 cm)\nx = 0')
        # self.lines_ifo, = self.ax.plot(np.zeros(76), np.zeros(76), label='Ifo, isat R1 prof\nz = 1054.35 cm')
        self.ax.plot([], [], label='Saved', linestyle='dashed', color='black')
        self.saved_lines_R1 = self.ax.plot(np.zeros(76), np.zeros(76), color='tab:blue', linestyle='dashed', alpha=0.7)[0]
        self.saved_lines_R4 = self.ax.plot(np.zeros(76), np.zeros(76), color='tab:orange', linestyle='dashed', alpha=0.7)[0]
        # self.saved_lines_ifo = self.ax.plot(np.zeros(76), np.zeros(76), color='tab:green', linestyle='dashed', alpha=0.7)[0]
        self.saved_text = self.ax.text(0, -1, '')

        self.ax_prof = self.fig.add_subplot(212)
        self.ax_prof.set_xlabel("Positions (cm)")
        self.ax_prof.set_ylabel(r"Isat (cm$^{-3}$)")
        self.ax_prof.set_xlim(-50, 50)
        self.ax_prof.set_ylim(-3e11, 1.3e13)

        self.prof_R1, = self.ax_prof.plot(np.zeros(1), np.zeros(1), label='t = 16')
        self.prof_R4, = self.ax_prof.plot(np.zeros(1), np.zeros(1), label='t = 16')
        # self.lines_ifo, = self.ax_prof.plot(np.zeros(1), np.zeros(1), label='Ifo, isat R1 prof\nz = 1054.35 cm')
        # self.ax_prof.plot([], [], label='Saved', linestyle='dashed', color='black')
        self.saved_prof_R1 = self.ax_prof.plot(np.zeros(1), np.zeros(1), color='tab:blue', linestyle='dashed', alpha=0.7)[0]
        self.saved_prof_R4 = self.ax_prof.plot(np.zeros(1), np.zeros(1), color='tab:orange', linestyle='dashed', alpha=0.7)[0]
        # self.saved_lines_ifo = self.ax_prof.plot(np.zeros(1), np.zeros(1), color='tab:green', linestyle='dashed', alpha=0.7)[0]

        self.ax.legend()
        self.ax_prof.legend()
        self.fig.tight_layout()
        self.navi = tkagg.NavigationToolbar2Tk(self.canvas, temp)
        self.navi.update()
        temp.pack(side=tk.RIGHT, expand=tk.TRUE, fill=tk.BOTH)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, pady=20, padx=(0, 10))
        self.navi.pack(side=tk.BOTTOM)

    def update_plot(self):
        # Update after every few seconds or so
        # print("==== updating plot")

        try:
            if self.data_q.empty() is False:
                # print("==== queue not empty")
                while not self.data_q.empty():
                    print('.', end='', flush=True)
                    self.msi = pickle.loads(self.data_q.get())

                self.data_recep_1 = process_msi(self.msi, self.pos_recep_1)
                self.data_recep_4 = process_msi(self.msi, self.pos_recep_4)
                self.data_recep_1 = concat_normalize_inputs(self.data_recep_1, self.max_values)
                self.data_recep_4 = concat_normalize_inputs(self.data_recep_4, self.max_values)
                self.predict_recep_1 = self.model(torch.tensor(self.data_recep_1)).detach().numpy()
                self.predict_recep_4 = self.model(torch.tensor(self.data_recep_4)).detach().numpy()

                self.predict_recep_1 *= self.max_values['isat']
                self.predict_recep_4 *= self.max_values['isat']

                Te = 5
                mult = 1 / ((0.81e-6 / 4 + np.pi * 0.5e-3 * 1e-3) *
                            np.exp(-0.5) * 1.602e-19 *
                            np.sqrt(1.602e-19 * Te / 6.646e-27))
                self.predict_recep_1 *= mult * 2 / 1e6
                self.predict_recep_4 *= mult * 2 / 1e6

                normed_prof = self.predict_recep_1 / np.sum(self.predict_recep_1, axis=0, keepdims=True)
                self.ifo = downsample(self.msi['interferometer_signals'][1, :])[0, 25:][np.newaxis, :]
                self.ifo = self.ifo * normed_prof * 1.705e+13  # n_bar_l

                # Index 15 is x=0
                self.timestamp = datetime.datetime.fromtimestamp(self.msi['diode_t0_seconds'][0] +
                                                                 self.msi['diode_t0_fraction'][0].astype('u8') / 2 ** 64)
                self.timestamp = self.timestamp - relativedelta(years=66, leapdays=-1)

                self.lines_R1.set_xdata(np.arange(76) / 2.5)
                self.lines_R1.set_ydata(self.predict_recep_1[15, :])
                self.lines_R4.set_xdata(np.arange(76) / 2.5)
                self.lines_R4.set_ydata(self.predict_recep_4[15, :])
                # self.lines_ifo.set_xdata(np.arange(76) / 2.5 * 2.4 / 4)
                # self.lines_ifo.set_ydata(self.ifo[15, :])
                self.prof_R1.set_xdata(self.x_loc)
                self.prof_R1.set_ydata(self.predict_recep_1[:, 40])
                self.prof_R4.set_xdata(self.x_loc)
                self.prof_R4.set_ydata(self.predict_recep_4[:, 40])

                # plt.draw()
                self.ax.set_title("Predicted profiles (T = {})".format(self.timestamp.strftime("%Y-%m-%d %H:%M:%S")))
                self.canvas.draw()
                gc.collect()  # to prevent canvas.draw from leaking memory
        except Exception as e:
            print("Exception: ")
            print(repr(e))
        self.after(100, self.update_plot)

    def save_trace(self):
        self.saved_R1 = self.predict_recep_1[15, :]
        self.saved_R4 = self.predict_recep_4[15, :]
        # self.saved_ifo = self.ifo[15, :]
        self.saved_p_R1 = self.predict_recep_1[:, 40]
        self.saved_p_R4 = self.predict_recep_4[:, 40]

        self.saved_lines_R1.set_xdata(np.arange(len(self.saved_R1)) / 2.5)
        self.saved_lines_R1.set_ydata(self.saved_R1)
        self.saved_lines_R4.set_xdata(np.arange(len(self.saved_R4)) / 2.5)
        self.saved_lines_R4.set_ydata(self.saved_R4)
        # self.saved_lines_ifo.set_xdata(np.arange(len(self.saved_ifo)) / 2.5 * 2.4 / 4)
        # self.saved_lines_ifo.set_ydata(self.saved_ifo)

        self.saved_prof_R1.set_xdata(self.x_loc)
        self.saved_prof_R1.set_ydata(self.saved_p_R1)
        self.saved_prof_R4.set_xdata(self.x_loc)
        self.saved_prof_R4.set_ydata(self.saved_p_R4)

        self.saved_text.set_text("Saved: T = {}".format(self.timestamp.strftime("%Y-%m-%d %H:%M:%S")))

    def clear_trace(self):
        self.saved_lines_R1.set_xdata([])
        self.saved_lines_R1.set_ydata([])
        self.saved_lines_R4.set_xdata([])
        self.saved_lines_R4.set_ydata([])
        # self.saved_lines_ifo.set_xdata([])
        # self.saved_lines_ifo.set_ydata([])
        self.saved_prof_R1.set_xdata([])
        self.saved_prof_R1.set_ydata([])
        self.saved_prof_R4.set_xdata([])
        self.saved_prof_R4.set_ydata([])
        self.saved_text.set_text('')

    def create_button(self, text, command, width=None, location=None, pady=0, size=9):
        '''Creates and packs a button.'''
        button = tk.Button(master=self.root, text=text, command=command,
                           font=('Helvetica', size), width=width)
        button.pack(side=location, pady=pady, padx=10, ipadx=10)


if __name__ == '__main__':
    root = tk.Tk()
    myapp = App(root)
    root.geometry("1024x640")
    root.title("Profile prediction (neural net-based)")
    root.config(bg='#345')
    myapp.mainloop()
