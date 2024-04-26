import time
import socket
import struct
import multiprocessing as mp
import numpy as np
import datetime
import tables
import datetime
import pickle
from pyphantom import Phantom, utils, cine


# Shots to save using the FFC. Do not choose shots immediately after probe movements!
# e.g., if you're taking 10 shots per position, then do not select shot 1, 11, or 21 etc..
#   because that would cause many cines to be taken for the specific shot (limitation of 
#   the datarun sequencer). The camera needs to set the proper settings before the shot
#   is taken.
FFC_shot_list = []  # shot list for the high-resolution cines
shots_per_position = 6  # please set to the proper value! This is for low-res cines
base_save_path = "G:\\saved_MSI\\Datarun_2024-4-14\\cines\\"  # change to some place with a lot of storage, end with a '\\'


def record_cine(cam):
    print("Recording... ", end='\r')
    # Clear all cines before recording
    cam.record(cine=1, delete_all=True)
    while(cam.partition_recorded(1) is False):
        print("Recording... waiting for cine  ", end='\r')
        time.sleep(0.01)  # sleep for 10 ms until a cine is recorded
    print("Recording... done   \t\t\t\t")


def save_cine(cam, frame_range, filename):
    rec_cine = cine.Cine.from_camera(cam, 1)
    rec_cine.save_range = frame_range
    rec_cine.save_non_blocking(filename=filename)
    # rec_cine.save(filename=filename)

    while (rec_cine.save_percentage < 100):
        print("Saving {}%   ".format(rec_cine.save_percentage), end='\r')
        time.sleep(0.01)
    print("Saving -- done")
    rec_cine.close()


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


def record_FFC_footage(q):
    ph = Phantom()
    cam_count = ph.camera_count
    if cam_count == 0:
        print("No camera discovered; closing.")
        ph.close()
        quit()

    # Assumes only one camera on the network
    cam = ph.Camera(0)
    curr_shot = 0

    msi = pickle.loads(q.get())
    cine_folder = base_save_path + 'small\\' + msi['datarun_key'] + '\\'
    print("Recording small cines")
    # Set camera settings
    cam.resolution = (256, 256)
    cam.exposure = 14  # microseconds, I ssume
    cam.edr_exposure = 0
    cam.frame_rate = 2500
    cam.post_trigger_frames = 200  # Save range *must* be less than this or else it will save the entire cine!
    print("Set small cine parameters -- recording as many shots as possible")

    previous_shotnum=0

    while True:
        record_cine(cam)

        while q.empty() is False:  # make sure that we don't get too far behind in shotnumber
            msi = pickle.loads(q.get())
        cine_folder = base_save_path + 'small\\' + msi['datarun_key'] + '\\'
        datarun_shotnum = msi['datarun_shotnum']
        ts_sec = datetime.datetime.fromtimestamp(msi['diode_t0_seconds'][0])
        ts_frac = msi['diode_t0_fraction'][0].astype('u8') / 2 ** 64
        ts_datarun = datetime.datetime.fromtimestamp(int(msi['datarun_timestamp']))
        ts_datarun_frac = np.modf(msi['datarun_timestamp'])[0]
        if msi['datarun_timeout'] == 1:
            print("MSI time: {}+{:0.4}".format(ts_sec, ts_frac), flush=True)
        else:
            print("Datarun shot: {}. Time: {}.{:0.4}. "
                  "MSI time: {}+{:0.4}".format(msi['datarun_shotnum'], ts_datarun,
                                                       ts_datarun_frac, ts_sec, ts_frac), flush=True)

        # if msi['datarun_timeout'] == 0 and (datarun_shotnum % shots_per_position) % 2 == 0:  # save every other shot
        if msi['datarun_timeout'] == 0:  # save every shot
            msi_time_string = "{}_{:0.4}".format(ts_sec, ts_frac)
            msi_time_string = "-".join(msi_time_string.split(':'))
            msi_time_string = "-".join(msi_time_string.split('.'))
            filename = 'shot-' + f'{datarun_shotnum:05}' + '_msi-' + msi_time_string
            filename = cine_folder + filename
            filename = "-".join(filename.split())
            print("Saving to " + filename, flush=True)
            # if previous_shotnum != datarun_shotnum:
            save_cine(cam, utils.FrameRange(-30, 160), filename)
            # else:
                # print("Same shotnumber -- skipping")
            # previous_shotnum = datarun_shotnum
        # else:
        #     time.sleep(1)



        # if curr_shot == msi['datarun_shotnum']:
        #     time.sleep(0.1)
        # else:
        #     curr_shot = msi['datarun_shotnum']

        #     # If datarun is ongoing and the datarun shotnumber is right before one we want:
        #     if msi['datarun_timeout'] == 0 and (datarun_shotnum + 1) in FFC_shot_list:
        #         cine_folder = base_save_path + '' + msi['datarun_key'] + '\\'
        #         print("Recording large cine")
        #         # Set camera settings
        #         cam.resolution = (256, 256)
        #         cam.exposure = 14  # microseconds, I ssume
        #         cam.edr_exposure = 0
        #         cam.frame_rate = 35000
        #         cam.post_trigger_frames = 2000  # Save range *must* be less than this or else it will save the entire cine!
        #         print("Set large cine parameters")
        #         record_cine(cam)

        #         msi = pickle.loads(q.get())
        #         datarun_shotnum = msi['datarun_shotnum']
        #         ts_sec = datetime.datetime.fromtimestamp(msi['diode_t0_seconds'][0])
        #         ts_frac = msi['diode_t0_fraction'][0].astype('u8') / 2 ** 64
        #         msi_time_string = "{}_{:0.4}".format(ts_sec, ts_frac)
        #         msi_time_string = "-".join(msi_time_string.split(':'))
        #         msi_time_string = "-".join(msi_time_string.split('.'))
        #         filename = 'shot-' + f'{datarun_shotnum:05}' + '_msi-' + msi_time_string
        #         filename = cine_folder + filename
        #         filename = "-".join(filename.split())
        #         print("Saving to " + filename, flush=True)
        #         save_cine(cam, utils.FrameRange(-300, 1600), filename)

        #     elif msi['datarun_timeout'] == 0 and (datarun_shotnum + 1) % shots_per_position == 2:
        #         cine_folder = base_save_path + 'small\\' + msi['datarun_key'] + '\\'
        #         print("Recording small cine")
        #         # Set camera settings
        #         cam.resolution = (256, 256)
        #         cam.exposure = 14  # microseconds, I ssume
        #         cam.edr_exposure = 0
        #         cam.frame_rate = 2500
        #         cam.post_trigger_frames = 200  # Save range *must* be less than this or else it will save the entire cine!
        #         print("Set small cine parameters -- recording four shots")
        #         record_cine(cam)

    #             msi = pickle.loads(q.get())
    #             datarun_shotnum = msi['datarun_shotnum']
    #             ts_sec = datetime.datetime.fromtimestamp(msi['diode_t0_seconds'][0])
    #             ts_frac = msi['diode_t0_fraction'][0].astype('u8') / 2 ** 64
    #             msi_time_string = "{}_{:0.4}".format(ts_sec, ts_frac)
    #             msi_time_string = "-".join(msi_time_string.split(':'))
    #             msi_time_string = "-".join(msi_time_string.split('.'))
    #             filename = 'shot-' + f'{datarun_shotnum:05}' + '_msi-' + msi_time_string
    #             filename = cine_folder + filename
    #             filename = "-".join(filename.split())
    #             print("Saving to " + filename, flush=True)
    #             save_cine(cam, utils.FrameRange(-30, 160), filename)

        #     else:
        #         pass

        
        # cine_folder = base_save_path + 'cines\\small\\' + msi['datarun_key'] + '\\'
        # filename = 'shot-' + f'{datarun_shotnum:05}' + '_msi-' + msi_time_string
        # filename = cine_folder + filename
        # filename = "-".join(filename.split())
        # print(filename)

    cam.close()
    ph.close()
    print("Done")


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

    ffc_process = mp.Process(target=record_FFC_footage, args=(q,))
    ffc_process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupting processes...")
    finally:
        multicast_process.terminate()
        multicast_process.join()
        ffc_process.terminate()
        ffc_process.join()
        multicast_process.close()
        ffc_process.close()
