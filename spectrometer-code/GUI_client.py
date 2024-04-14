#
# Creates GUI with tkinter in combination with methods from client.py.
#

# For GUI
from tkinter.constants import BOTTOM, CENTER, LEFT, TOP, RIGHT, HORIZONTAL
import tkinter as tk
from tkinter.ttk import Progressbar

# Client code
from client import SendSettings, ReceiveSpecData
import socket
import multiprocessing as mp

# For plotting in GUI
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# uhh Other stuff
import numpy as np
import gc
import struct
import queue

# from memory_profiler import profile

# @profile(precision=6)
def data_client_process(pipe, data_queue):
    # This will raise an EOFError if the pipe is closed from the other end while waiting
    # Theoretically the first time this is ran, do_connect should be True
    while True:  # should be "while alive" where alive is some shared variable
        do_connect = pipe.recv()
        if do_connect == "quit":
            return
        if do_connect is False:
            continue
        print("Connecting to data server...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(do_connect)
            sock.connect(do_connect)
            print("Connected to data server. ")

            while do_connect is not False and do_connect != "quit":  # while connected...
                # Get trigger value
                trigger = struct.unpack(">i", sock.recv(4))[0]
                # Get integration time
                integration_time = struct.unpack(">i", sock.recv(4))[0]
                # Get data size
                data_length = struct.unpack(">i", sock.recv(4))[0]

                data = b''
                data_remaining = data_length

                BUFF_LEN = 1024
                while data_remaining > 0:
                    part = sock.recv(BUFF_LEN if data_remaining > BUFF_LEN else data_remaining)
                    data += part
                    data_remaining -= len(part)
                spec_data = np.frombuffer(data, dtype=np.float64).reshape(2, -1)
                data_queue.put((trigger, integration_time, spec_data))

                if pipe.poll() is True:
                    do_connect = pipe.recv()
            sock.close()
            if do_connect == "quit":
                return

        except Exception as e:
            print(e)


# Send settings to the spectroemter server
# @profile(precision=6)
def send_settings_process(pipe, settings_queue):
    # This will raise an EOFError if the pipe is closed from the other end while waiting
    # Theoretically the first time this is ran, do_connect should be True
    while True:  # should be "while alive" where alive is some shared variable
        do_connect = pipe.recv()
        if do_connect == "quit":
            return
        if do_connect is False:
            continue
        print("Connecting to settings server...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(do_connect)
            sock.connect(do_connect)
            print("Connected to settings server. ")

            while do_connect is not False and do_connect != "quit":  # while connected...
                # Send settings. Use try-except block here so that settings_queue can block
                try:
                    # print("Checking settings queue...")
                    settings = settings_queue.get(timeout=0.2)
                    sock.send(struct.pack(">i", settings[0]))  # Send trigger value
                    sock.send(struct.pack(">i", settings[1]))  # Send integration time
                    print("Sent settings: trigger {}, integration {}".format(settings[0], settings[1]))
                except queue.Empty:
                    pass

                if pipe.poll() is True:
                    do_connect = pipe.recv()
            sock.close()
            if do_connect == "quit":
                return

        except Exception as e:
            print(e)


class SpecApp(tk.Frame, SendSettings, ReceiveSpecData):
    '''Creates GUI.

    Attributes
    ----------
    root (class 'tkinter.Tk)
    '''
    # @profile(precision=6)
    def __init__(self, root):
        '''Constructor method for App class.

        Creates setting entries, a quit button, and a plot for the GUI.

        Parameters
        ----------
        root: <class 'tkinter.Tk'>
        '''
        # Initializing tkinter Frame class.
        tk.Frame.__init__(self, root)
        # Initializing SendParameters class from client.py
        # SendSettings.__init__(self, server_address=(), trig_val=0, int_time_micros=0)
        # Initializing Receive class from client.py
        # ReceiveSpecData.__init__(self)
        # Initializing SaveData class
        # SaveData.__init__(self, file_number=0)

        # Not sure what this is, but it packs the GUI frame or something
        self.root = root
        self.pack()

        # Initializing some things
        self.spec_list = []
        self.file_number = 0
        self.save_bool = True

        # tells program that there is no re-entry yet
        self.re_entry = False

        # Numpy array for storing all the spectrometer info
        self.data_list = []

        # Handle all the connections and multiprocessing stuff
        # Make queue for recieving the data from the data client process
        self.data_queue = mp.Queue()
        # Pipe to tell the thread when to start
        self.connect_pipe_parent, self.connect_pipe_child = mp.Pipe()
        self.data_client = mp.Process(target=data_client_process,
                                      args=(self.connect_pipe_child, self.data_queue))
        self.data_client.start()
        self.data_connected = False

        # Make queue for sending the spectrometer settings
        self.settings_queue = mp.Queue()
        # Pipe to tell the thread when to start
        self.settings_pipe_parent, self.settings_pipe_child = mp.Pipe()
        self.settings_client = mp.Process(target=send_settings_process,
                                          args=(self.settings_pipe_child, self.settings_queue))
        self.settings_client.start()
        self.settings_connected = False

        # Creates the spectrum plot
        self.create_plot()
        self.update_plot()

        # creates entries with their label
        self.shot_avg = self.create_entry("Shot average", default=1, pady=(30, 0))
        self.trig_entry = self.create_entry("Trigger Value", default=3, pady=(5, 0))
        self.trig_entry.config(state="disabled")
        self.int_entry = self.create_entry("Integration Time (\u03BCs)", default=20000, pady=(5, 0))
        self.int_entry.config(state="disabled")
        self.ip_entry = self.create_entry("Server IP", default='192.168.7.91', pady=(5, 0))
        self.port_entry = self.create_entry("Data Port", default=5004, pady=(5, 0))
        self.control_port_entry = self.create_entry("Control Port", default=5005, pady=(5, 0))
        # self.setting_entries = [trig_entry, int_entry, ip_entry, port_entry, control_port_entry]

        # Creates all the buttons
        # self.create_button("Start Only\nData Collection", self.collect_only_data, location=TOP, pady=(10, 5), width=15)
        # self.create_button("Start Animation", self.display_animation,
        #                    location=TOP, pady=(10, 0), width=15)
        # self.create_button("Connect to data server", self.connect_to_data_server, location=TOP, pady=(10, 0), width=15)

        self.data_connect_button = tk.Button(master=self.root, text="Connect to data server",
                                             command=self.connect_to_data_server,
                                             font=('Helvetica', 9), width=20)
        self.data_connect_button.pack(side=TOP, pady=(10, 0), padx=10, ipadx=0)

        self.settings_connect_button = tk.Button(master=self.root, text="Connect to settings server",
                                                 command=self.connect_to_settings_server,
                                                 font=('Helvetica', 9), width=20)
        self.settings_connect_button.pack(side=TOP, pady=(10, 0), padx=10, ipadx=0)

        self.send_settings_button = tk.Button(master=self.root, text="Send settings",
                                              command=self.send_settings,
                                              font=('Helvetica', 9), width=20, state="disabled")
        self.send_settings_button.pack(side=TOP, pady=(10, 0), padx=10, ipadx=0)

        self.create_button("Quit", self._quit, location=BOTTOM, pady=20)
        # self.create_button("Save Most Recent", self.save_spectrum_files,
        #                    location=TOP, pady=(10, 10), width=15)

        # Creates an empty progress bar.
        # self.create_progress_bar()
        # Creates the percentage under the progress bar.
        # self.progress_label = self.create_label(text='0%')

        # Creates notes.
        self.create_text(
            "------------- Notes ------------\n" +
            "Trigger Modes:\nnormal = 0\nlevel/software = 1\nsynchronization = 2\nedge/hardware = 3\n" +
            "\nIntegration Time Range:\n3.8ms-10s")
    # @profile(precision=6)
    def connect_to_data_server(self):
        if self.data_connected is False:
            self.data_connected = True
            self.connect_pipe_parent.send((self.ip_entry.get(), int(self.port_entry.get())))
            self.data_connect_button.config(text="Disconnect from data server")
            print("connect button pressed")
        elif self.data_connected is True:
            self.data_connected = False
            self.connect_pipe_parent.send(False)
            self.data_connect_button.config(text="Connect to data server")
            print("disconnect button pressed")
    # @profile(precision=6)
    def connect_to_settings_server(self):
        if self.settings_connected is False:
            self.settings_connected = True
            self.settings_pipe_parent.send((self.ip_entry.get(), int(self.control_port_entry.get())))
            self.settings_connect_button.config(text="Disconnect from settings server")
            self.send_settings_button.config(state="normal")
            self.trig_entry.config(state="normal")
            self.int_entry.config(state="normal")
            print("settings connect button pressed")
        elif self.settings_connected is True:
            self.settings_connected = False
            self.settings_pipe_parent.send(False)
            self.settings_connect_button.config(text="Connect to settings server")
            self.send_settings_button.config(state="disabled")
            self.trig_entry.config(state="disabled")
            self.int_entry.config(state="disabled")
            print("settings disconnect button pressed")

    def send_settings(self):
        self.settings_queue.put((int(self.trig_entry.get()), int(self.int_entry.get())))
        print("Send button pressed")

    def _quit(self):
        '''Closes the window and stops the mainloop.'''

        # Close the data client connection
        self.connect_pipe_parent.send("quit")
        self.data_client.join(timeout=0.5)
        self.data_client.close()
        # Close the settings connection
        self.settings_pipe_parent.send("quit")
        self.settings_client.join(timeout=0.5)
        self.settings_client.close()

        self.root.quit()     # stops mainloop
        self.root.destroy()

    def create_progress_bar(self):
        '''Creates a progress bar that progresses when saving.'''
        self.progress = Progressbar(self.root, orient=HORIZONTAL, length=125, mode='determinate')
        self.progress.pack(padx=20, side=TOP)

    def create_button(self, text: str, command, width=None, location=None, pady=0, size=9):
        '''Creates and packs a button.'''
        button = tk.Button(master=self.root, text=text, command=command,
                           font=('Helvetica', size), width=width)
        button.pack(side=location, pady=pady, padx=10, ipadx=10)
    # @profile(precision=6)
    def create_plot(self):
        '''Creates and packs embedded plot.'''
        temp = tk.LabelFrame(self.root)

        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Wavelengths")
        self.ax.set_ylabel("Intensities")
        self.ax.set_xlim(275, 775)
        self.ax.set_ylim(2.5e2, 2e4)
        self.ax.set_yscale('log')
        # self.ax.set_title("Spectrum")
        self.ax.set_title("Spectrum ({} shot avg) -- trigger: {}, integration: {} $\\mu$s".format(0, 0, 0))
        self.canvas = FigureCanvasTkAgg(self.fig, master=temp)
        self.lines, = self.ax.plot(np.zeros(3648), np.zeros(3648))
        self.navi = tkagg.NavigationToolbar2Tk(self.canvas, temp)
        self.navi.update()
        temp.pack(side=tk.RIGHT, expand=tk.TRUE, fill=tk.BOTH)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, pady=20, padx=(0, 10))
        self.navi.pack(side=tk.BOTTOM)
    # @profile(precision=6)
    def update_plot(self):
        # Update after every few seconds or so
        # print("==== updating plot")
        try:
            if self.data_queue.empty() is False:
                # print("==== queue not empty")
                while not self.data_queue.empty():
                    trigger, integration_time, spec_data = self.data_queue.get()
                    self.data_list.append(spec_data)
                    while len(self.data_list) > 1024:
                        self.data_list.pop(0)
                self.data_array = np.array(self.data_list)
                num_shots = int(self.shot_avg.get())
                num_shots = num_shots if num_shots <= self.data_array.shape[0] else self.data_array.shape[0]
                avg_spectrum = np.mean(self.data_array[-num_shots:], axis=0)

                self.lines.set_xdata(avg_spectrum[0])
                self.lines.set_ydata(avg_spectrum[1])
                # plt.draw()
                self.ax.set_title("Spectrum ({} shot avg) -- trigger: {}, integration: {} $\\mu$s".format(num_shots, trigger, integration_time))
                self.canvas.draw()
                gc.collect()  # to prevent canvas.draw from leaking memory
        except Exception as e:
            print(e)
        self.after(100, self.update_plot)

    def create_entry(self, entry: str, default='', pady=0) -> tk.Entry:
        '''Creates an entry for an `int` variable.

        Parameters
        ----------
        entry: <str>
            Name of label.
        default: <int> or <str>
            Default value for entries.

        Returns
        -------
        entry_thingy: <class 'tkinter.Entry'>
        '''
        self.create_label(text=entry, pady=pady)  # Creates label
        self.entry_thingy = tk.Entry(width=20)  # Creates entry field
        self.entry_thingy.pack(pady=0, padx=30)  # Packs the entry
        self.contents = tk.IntVar()
        self.contents.set(default)  # Sets default for field
        self.entry_thingy["textvariable"] = self.contents

        # binding enter/return key to storing settings and animation
        # self.entry_thingy.bind('<Key-Return>', self.save_settings)

        return self.entry_thingy

    def create_label(self, text: str, pady=0, location=None) -> tk.StringVar:
        '''Creates labels.

        Returns
        -------
        var: <class tkinter.StringVar>
        '''
        self.var = tk.StringVar()
        self.var.set('{}'.format(text))

        self.label = tk.Label(self.root, textvariable=self.var)
        self.label.pack(side=location, pady=pady, padx=0)
        self.label.config(fg='#fff', bg='#345')

        return self.var

    def create_text(self, text_parameter: str):
        '''Creates a text box.

        Parameters
        ----------
        text_parameter: <str>
            creates text in GUI.
        '''
        text = tk.Text(self.root, height=5, width=20, font=('Helvetica', 9))
        text.pack(pady=(20, 0), fill=tk.Y, expand=1)
        text.insert(tk.END, '{}'.format(text_parameter))
        text.config(fg='#fff', bg='#345')

    # def save_spectrum_files(self):
    #     '''Saves both npz and mp4 files.'''
    #     self.save_bool = False
    #     self.ani.event_source.stop()
    #     d = SaveData(self.file_number)
    #     # Saves the data as numpy array.
    #     d.save_data(self.spec_list)
    #     npzfiles = d.load_data()

    #     amount_of_spectra = len(npzfiles['arr_0'])

    #     # annoying to take this out of save_spectrum_files method so it's staying for now
    #     def animate_for_saving(i):
    #         '''Re-creates animation with saved data and updates progress bar.'''
    #         wavelengths = npzfiles['arr_0'][0][0]
    #         intensities = npzfiles['arr_0'][i][1]

    #         self.ax.cla()
    #         self.ax.set_xlabel("Wavelengths")
    #         self.ax.set_ylabel("Intensities")
    #         self.ax.set_title("Spectrum {} / {}".format(i + 1, amount_of_spectra))
    #         self.ax.plot(wavelengths, intensities)

    #         percentage = (i + 1) / len(npzfiles['arr_0']) * 100
    #         self.progress['value'] = percentage
    #         self.progress_label.set("{:.1f}%".format(percentage))
    #         self.root.update_idletasks()

    #         gc.collect()

    #     d.save_animation_file(self.fig, animate_for_saving, amount_of_spectra)
    #     self.spec_list.clear()

    #     self.file_number += 1
    #     self.save_bool = True
    #     self.ani.event_source.start()

    # def save_settings(self):
    #     '''Stores user's inputs into a list and uses those inputs to send and receive data, which
    #     is used to create an animation.
    #     '''
    #     setting_list = []

    #     # goes through length of setting_entries and appends entries to list
    #     for self.contents in self.setting_entries:
    #         setting_list.append(self.contents.get())

    #     # settings for spectrometer
    #     trig_val = int(setting_list[0])
    #     self.int_time_micros = int(setting_list[1])
    #     self.server_address = (setting_list[2], int(setting_list[3]))

    #     self.send_settings_from_gui(self.server_address, trig_val, self.int_time_micros)

    # def send_settings_from_gui(self, server_address, trig_val, int_time_micros):
    #     '''Sends settings from GUI using methods from SendSettings

    #     Parameters
    #     ----------
    #     server_address: <tuple>
    #         Contains ip address (group) and port number in form (group, port).
    #     trig_val: <int>
    #         Sets trigger mode:
    #             normal = 0,
    #             level/software = 1,
    #             synchronization = 2,
    #             edge/hardware = 3
    #     int_time_micros: <int>
    #         Integration time in microseconds.
    #     '''
    #     self.s = SendSettings(server_address, trig_val, int_time_micros)
    #     self.s.send_settings()
    #     self.client_port = self.s.get_port()

    # def display_animation(self):
    #     '''Creates the animation.'''
    #     self.save_settings()
    #     # Initializing class that receives spectrum data.
    #     self.r = ReceiveSpecData()

    #     # Creates animation
    #     interval = self.int_time_micros / 1000
    #     self.ani = FuncAnimation(self.fig,
    #                              self.animate,
    #                              fargs=(self.s, self.r, self.server_address, self.client_port,),
    #                              interval=0,
    #                              cache_frame_data=False)

    #     self.canvas.draw()
    #     self.more_than_one_entry(self.ani)

    # def more_than_one_entry(self, ani):
    #     '''Prevents from multiple animations running at once.

    #     Stops the current animation instance so there is never more than 1 animation running. 
    #     Usually activated after saving more than once or changing the settings

    #     Parameters
    #     ----------
    #     ani: <class 'FuncAnimation'>
    #         Animates data.
    #     '''
    #     # prevents from multiple animations running at once
    #     if self.re_entry == True:
    #         ani.event_source.stop()

    #     self.re_entry = True

    # def animate(self, i, send, receive, server_address, client_port):
    #     '''Continuously receives data from server and plots it.

    #     Used to animate spectrum received from spectrometer. Uses methods from the `SendSettings` and
    #     `ReceiveSpecData` classes. Every time the method loops around, it sends zero bytes to the server to indicate
    #     that it should continue sending data.

    #     Parameters
    #     ----------
    #     i: <int>
    #         Needed for FuncAnimation.
    #     send: <class 'SendSettings'>
    #         class from client.py that sends spectrometer settings.
    #     receive: <class 'ReceiveSpecData'>
    #         class from client.py that receives spectrum data.
    #     server_address: <list>
    #         List in the form of [IP, port]. The IP and port are the server's IP and port.
    #     client_port: <int>
    #         This is the port that the client has been assigned (after sending something) to receive data.

    #     Notes
    #     -----
    #     Having `send` and `receive` as parameters is really odd but I haven't found a solution to it yet.
    #     '''
    #     spec = self.receive_data_for_gui(send, receive, server_address, client_port)
    #     # Clearing and then plotting
    #     self.ax.cla()
    #     self.ax.set_xlabel("Wavelengths")
    #     self.ax.set_ylabel("Intensities")
    #     self.ax.set_title("Spectrums")
    #     self.ax.plot(spec[0], spec[1])
    #     gc.collect()

    # def receive_data_for_gui(self, send, receive, server_address, client_port):
    #     '''Does all the receiving using methods from client.py.

    #     Parameters
    #     ----------
    #     i: <int>
    #         Needed for FuncAnimation.
    #     send: <class 'SendSettings'>
    #         class from client.py that sends spectrometer settings.
    #     receive: <class 'ReceiveSpecData'>
    #         class from client.py that receives spectrum data.
    #     server_address: <list>
    #         List in the form of [IP, port]. The IP and port are the server's IP and port.
    #     client_port: <int>
    #         This is the port that the client has been assigned (after sending something) to receive data.

    #     Notes
    #     -----
    #     Having `send` and `receive` as parameters is really odd but I haven't found a solution to it yet.
    #     '''
    #     # Sends zero bytes to server so the server knows when to continue sending data.
    #     send.send_zero_bytes()
    #     receive.create_socket()

    #     # Receiving data
    #     receive.set_socket_receive()
    #     receive.bind_socket(client_port, server_address)
    #     receive.receive_data()
    #     spec = receive.prepare_data()
    #     self.spec_list.append(spec)
    #     return spec

# class SaveData:
#     '''Class that takes care of saving data.

#     Attributes
#     ----------
#     file_number: <int>
#         determines which file number you are on when saving.

#     Notes
#     -----
#     Have to be careful when closing out GUI then opening it up again because file_number restarts of course.
#     '''

#     def __init__(self, file_number):
#         '''Constructor for SaveData class.

#         Parameters
#         ----------
#         file_number: <int>
#             determines which file number you are on when saving.
#         '''
#         self.file_number = file_number
#         self.file_name = 'C:/Users/orrk9/Code/spectrometer/servers/testing_saving copy 2/animations_and_npz/spectrums_run00{}'.format(
#             file_number)

#     def save_data(self, spec):
#         '''Saves raw data of a spectrum in a .npz file.

#         Parameters
#         ----------
#         spec: <numpy.array> or <list>
#             Spectrum that is being saved. 2D array or list.
#         '''
#         np.savez(self.file_name + ".npz", spec)

#     def load_data(self):
#         '''Loads the .npz file.'''
#         npzfiles = np.load(self.file_name + ".npz")
#         npzfiles.files
#         return npzfiles

#     def save_animation_file(self, fig, save_ani, amount_of_spectra):
#         '''Saves an animation of the saved spectrum data in .mp4 format.

#         Parameters
#         ----------
#         fig: <class matplotlib.pyplot Figure>
#         save_ani: <class FuncAnimation>
#             Animation function that creates the data that you want to save.
#         amount_of_spectra: <int>
#              The number of spectra that is being saved (aka the amount of frames in your animation).
#         '''
#         ani_save = FuncAnimation(fig,
#                                  save_ani,
#                                  cache_frame_data=False,
#                                  frames=amount_of_spectra)

#         # Saves the animation as .mp4 file.
#         ani_save.save(self.file_name + ".mp4")

#         ani_save.event_source.stop
