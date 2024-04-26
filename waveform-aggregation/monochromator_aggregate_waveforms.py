import time
import datetime
import tables
import glob
from tqdm import tqdm
from scipy.signal import hilbert
from scipy import constants as const
import re
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import pyformulas as pf

from oscilloscope_helper import ReadBinaryTrace, Trace, float2eng


# ----- ASSUMES 10 MHz SAMPLING FREQUENCY FOR 1 MS ----- #
if __name__ == '__main__':
    # fig = plt.figure(figsize=(10, 8))
    # canvas = np.zeros((640, 960))
    # screen = pf.screen(canvas, "Line-integrated density")
    shots_saved = 0
    while(True):
        # trace_len = ReadBinaryTrace(file_list[0])[2].shape[0]
        # Set length of data arrays in table based on first file in list
        # Trace.time = tables.Float16Col(shape=(trace_len,))
        # Trace.y1 = tables.Float16Col(shape=(trace_len,))

        time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")

        h5_file = tables.open_file('G:saved_MSI\\Datarun_2024-4-14\\monochromator\\' + time_str + '.h5', 'w',
                                   title="Monochrometer-" + time_str)
        h5_group = h5_file.create_group("/", "lines", "Traces of spectral lines")
        h5_tableC1 = h5_file.create_table(h5_group, 'Channel1', Trace, 'Time traces from Channel 1')
        h5_tableC3 = h5_file.create_table(h5_group, 'Channel3', Trace, 'Time traces from Channel 3')
        h5_tableC4 = h5_file.create_table(h5_group, 'Channel4', Trace, 'Time traces from Channel 4')
        h5_rowC1 = h5_tableC1.row
        h5_rowC3 = h5_tableC3.row
        h5_rowC4 = h5_tableC4.row

        print("Saving to G:saved_MSI/Datarun_2024-4-14/monochromator/" + time_str + ".h5")

        total_shots = 0
        while(total_shots < 3600):
            file_list = sorted(glob.glob("G:saved_MSI\\Datarun_2024-4-14\\monochromator\\traces\\C4*.trc"))
            if file_list:
                plt.clf()
                for f in file_list:  # so that this can keep up
                    try:
                        # Read last channel first so that all waveforms are on disk
                        C4 = ReadBinaryTrace(f)
                        C3 = ReadBinaryTrace(re.sub(r'\\C4', r'\\C3', f))
                        C1 = ReadBinaryTrace(re.sub(r'\\C4', r'\\C1', f))
                        # print(ref[3], signal[3])

                        # Downsample to 10,000 data points (original: 10 MHz sampling frequency for 0.1s)
                        # If sampling at 100 MHz, this will downsample to a 100 kHz sampling rate

                        print("Time: {}".format(C1[0]['TRIGGER_TIME']))
                        # times = np.mean(signal[2][:-2].reshape(-1, 100), axis=1)
                        # h5_row['times'] = times
                        h5_rowC1['time_trace'] = C1[3][:-2]
                        h5_rowC3['time_trace'] = C3[3][:-2]
                        h5_rowC4['time_trace'] = C4[3][:-2]
                        # print(density)
                        for k in C1[0].keys():
                            if k == 'TRIGGER_TIME':
                                h5_rowC1[k] = np.array(C1[0][k])
                                h5_rowC3[k] = np.array(C3[0][k])
                                h5_rowC4[k] = np.array(C4[0][k])
                            else:
                                h5_rowC1[k] = C1[0][k]
                                h5_rowC3[k] = C3[0][k]
                                h5_rowC4[k] = C4[0][k]
                        h5_rowC1.append()
                        h5_rowC3.append()
                        h5_rowC4.append()
                        total_shots += 1
                        shots_saved += 1

                        # plt.ion()
                        # plt.show()
                        # plt.title('Monochrometer traces at {}/{} {}:{}:{:.2f}'.format(
                        #     C1[0]['TRIGGER_TIME'][4],
                        #     C1[0]['TRIGGER_TIME'][3],
                        #     C1[0]['TRIGGER_TIME'][2],
                        #     C1[0]['TRIGGER_TIME'][1],
                        #     C1[0]['TRIGGER_TIME'][0]))
                        # plt.ylabel('Monochrometer traces')
                        # plt.xlabel('Time (ms)')
                        # offset = C1[0]['HORIZ_OFFSET']
                        # # Multiply by 1000 to get the time in ms instead of s
                        # timebase = C1[0]['HORIZ_INTERVAL']
                        # time_axis = (np.arange(100000) * timebase + offset) * 1000
                        # plt.plot(time_axis, C1[3][:-2], color=[1.0, 0.0, 0.0, 0.3])  # more like BGR
                        # plt.plot(time_axis, C3[3][:-2], color=[0.0, 1.0, 0.0, 0.3])  # more like BGR
                        # plt.plot(time_axis, C4[3][:-2], color=[0.0, 0.0, 1.0, 0.3])  # more like BGR
                        # plt.tight_layout()
                        # fig.canvas.draw()
                        # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        # screen.update(image)
                        # plt.draw()
                        # plt.pause(0.001)

                    except Exception as e:
                        print(e)
                    # print("+", end="")
                # print(" +{} ".format(len(file_list)), end="")
                h5_tableC1.flush()
                h5_tableC3.flush()
                h5_tableC4.flush()
                for f in file_list:
                    try:
                        os.remove(f)  # Remove channel 2
                        os.remove(os.path.normpath(re.sub(r'\\C4', r'\\C3', f)))  # Remove channel 1
                        os.remove(os.path.normpath(re.sub(r'\\C4', r'\\C1', f)))  # Remove channel 1
                    except Exception as e:
                        print(e)
                # print(" ({}) ".format(total_shots))
                file_list = []

            else:
                print(".", end="")
            time.sleep(10) # sleep before trying again
        h5_file.close()
