# import sys
# sys.path.append("/data/phil/lapd-msi/")
# import io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# import datetime
import tables  # pytables
# import scipy.signal as sig
# import bapsflib
import glob

plt.style.use(['dark_background'])
mpl.rcParams['axes.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
mpl.rcParams['figure.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
mpl.rcParams['legend.facecolor'] = (50 / 256, 50 / 256, 50 / 256)
mpl.rcParams['savefig.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
# data_color = (0.2, 0.7, 1.0, 1.0)
# samp_color = (1.0, 0.7, 0.1, 0.6)
# aux_color = (0, 0.9, 0.0, 0.6)

# path = "/data/phil/MSI/2021-12-24_onwards/"
# filenames = [f for f in glob.glob("Big*.h5")]
# filenames += [f for f in glob.glob("ICRF*.h5")]
# filenames += [f for f in glob.glob("Koepke*.h5")]
# filenames += [f for f in glob.glob("Nonlinear*.h5")]
# filenames += [f for f in glob.glob("Big*.h5")]
filenames = [
    'ICRF_campaign 1973 04_EBL_xyplane_HeAr_7E11peakdensity_2kG.h5',
#     'data_run_id 1953 ICRF_campaign.h5',
#     'data_run_id 1954 ICRF_campaign.h5'
]

# magnet_path = "/data/phil/MSI/Oct_datarun_Yhoshua/probe_data/" + \
#               "01_harpin_line_800G 2021-10-12 18.27.47.hdf5"
# magnet_profile_z = bapsflib.lapd.File(magnet_path).read_msi('magnetic field').info['z']

# Add time series over datarun images

for name in filenames:
    print("Plotting file " + name)
    msi_file = tables.open_file(name, 'r')
    msi_data = msi_file.root.MSI.data[::4]
    msi_file.close()

    disp_shots = 800 if msi_data.shape[0] > 800 else msi_data.shape[0]
    idx = np.linspace(0, msi_data.shape[0] - 1, disp_shots)
    idx = np.around(idx).astype(int)

    alpha = np.clip(0.05 * 800 / disp_shots, 0.0, 1.0)
    c_discharge = [0.3, 0.9, 1.0, alpha]
    c_interferometer = [0.2, 1.0, 0.3, alpha]
    c_diode = [1.0, 0.9, 0.3, alpha]
    c_white = [1.0, 1.0, 1.0, alpha]

    # 21"x12" for 16:9 aspect ratio on 25" monitor
    fig, ax = plt.subplots(3, 4, figsize=(50, 28))
    fig.suptitle(name[:-3])
    ax[0, 0].text(0.5, 0.95, 'Discharge current', color='w', transform=ax[0, 0].transAxes)
    ax[0, 0].plot(np.arange(msi_data['discharge_current'].shape[1])[np.newaxis, :].T,
                  msi_data['discharge_current'][idx, :].T,
                  color=c_discharge)

    ax[1, 0].text(0.5, 0.95, 'Discharge voltage', color='w', transform=ax[1, 0].transAxes)
    ax[1, 0].plot(np.arange(msi_data['discharge_voltage'].shape[1])[np.newaxis, :].T,
                  msi_data['discharge_voltage'][idx, :].T,
                  color=c_discharge)

    ax[2, 0].text(0.5, 0.95, 'Interferometer', color='w', transform=ax[2, 0].transAxes)
    ax[2, 0].plot(np.arange(msi_data['interferometer_signals'].shape[2])[np.newaxis, :].T,
                  msi_data['interferometer_signals'][idx, 0, :].T,
                  color=c_interferometer)

    ax[0, 1].text(0.5, 0.95, 'diode 0', color='w', transform=ax[0, 1].transAxes)
    ax[0, 1].plot(np.arange(msi_data['diode_signals'].shape[2])[np.newaxis, :].T,
                  msi_data['diode_signals'][idx, 0, :].T,
                  c=c_diode)

    ax[1, 1].text(0.5, 0.95, 'diode 1', color='w', transform=ax[1, 1].transAxes)
    ax[1, 1].plot(np.arange(msi_data['diode_signals'].shape[2])[np.newaxis, :].T,
                  msi_data['diode_signals'][idx, 1, :].T,
                  c=c_diode)

    ax[2, 1].text(0.5, 0.95, 'diode 2', color='w', transform=ax[2, 1].transAxes)
    ax[2, 1].plot(np.arange(msi_data['diode_signals'].shape[2])[np.newaxis, :].T,
                  msi_data['diode_signals'][idx, 2, :].T,
                  c=c_diode)

    try:
        ax[0, 2].text(0.5, 0.95, 'diode 3', color='w', transform=ax[0, 2].transAxes)
        ax[0, 2].plot(np.arange(msi_data['north_discharge_current'].shape[1])[np.newaxis, :].T,
                      msi_data['north_discharge_current'][idx, :].T,
                      c=c_diode)

        ax[1, 2].text(0.5, 0.95, 'diode 4', color='w', transform=ax[1, 2].transAxes)
        ax[1, 2].plot(np.arange(msi_data['north_discharge_voltage'].shape[1])[np.newaxis, :].T,
                      msi_data['north_discharge_voltage'][idx, :].T,
                      c=c_diode)
    except:
        print("No north discharge info")

    ax[2, 2].text(0.5, 0.95, 'Manget profile', color='w', transform=ax[2, 2].transAxes)
    ax[2, 2].plot(msi_data['magnet_profile'][idx, :].T,
                  c=c_white)

    ax[2, 3].text(0.5, 0.95, 'Gas pressures (log)', color='w', transform=ax[2, 3].transAxes)
    ax[2, 3].scatter(np.arange(51)[np.newaxis, :] + np.random.uniform(low=-0.4, high=0.4, size=(disp_shots, 1)),
                     np.concatenate((msi_data['pressure_fill'][idx, np.newaxis],
                                     msi_data['RGA_partials'][idx, :]), axis=1),
                     c=[c_white], s=0.7)
    # Fake plot ot get autoscaling to work correctly. Weird bug.
    ax[2, 3].plot(np.arange(51)[np.newaxis, :] + np.zeros((disp_shots, 1)),
                  np.concatenate((msi_data['pressure_fill'][idx, np.newaxis],
                                  msi_data['RGA_partials'][idx, :]), axis=1), alpha=0)
    ax[2, 3].set_yscale('log')

    plt.tight_layout()
    plt.savefig("summaries/" + name[:-3] + '-traces.png')
    plt.close()

    alpha = (0.05 if msi_data.shape[0] > 100000 else 1.0)
    # np.clip(0.05 * 800 / disp_shots, 0.0, 1.0)
    c_discharge = [0.3, 0.9, 1.0, alpha]
    c_interferometer = [0.2, 1.0, 0.3, alpha]
    c_diode = [1.0, 0.9, 0.3, alpha]
    c_white = [1.0, 1.0, 1.0, alpha]

    # Add plots that track over the course of the datarun / days
    x_axis = np.arange(msi_data.shape[0])

    fig, ax = plt.subplots(3, 4, figsize=(50, 28))
    fig.suptitle(name[:-3])
    ax[0, 0].text(0.5, 0.95, 'Discharge current', color='w', transform=ax[0, 0].transAxes)
    ax[0, 0].scatter(x_axis, np.mean(msi_data['discharge_current'], axis=1), s=1.0, c=[c_discharge])

    ax[1, 0].text(0.5, 0.95, 'Discharge voltage', color='w', transform=ax[1, 0].transAxes)
    ax[1, 0].scatter(x_axis, np.mean(msi_data['discharge_voltage'], axis=1), s=1.0, c=[c_discharge])

    ax[2, 0].text(0.5, 0.95, 'Interferometer', color='w', transform=ax[2, 0].transAxes)
    ax[2, 0].scatter(x_axis, np.mean(msi_data['interferometer_signals'][:, 0, :], axis=1), s=1.0, c=[c_interferometer])

    ax[0, 1].text(0.5, 0.95, 'diode 0', color='w', transform=ax[0, 1].transAxes)
    ax[0, 1].scatter(x_axis, np.mean(msi_data['diode_signals'][:, 0, :], axis=1), s=1.0, c=[c_diode])

    ax[1, 1].text(0.5, 0.95, 'diode 1', color='w', transform=ax[1, 1].transAxes)
    ax[1, 1].scatter(x_axis, np.mean(msi_data['diode_signals'][:, 1, :], axis=1), s=1.0, c=[c_diode])

    ax[2, 1].text(0.5, 0.95, 'diode 2', color='w', transform=ax[2, 1].transAxes)
    ax[2, 1].scatter(x_axis, np.mean(msi_data['diode_signals'][:, 2, :], axis=1), s=1.0, c=[c_diode])

    try:
        ax[0, 2].text(0.5, 0.95, 'diode 3', color='w', transform=ax[0, 2].transAxes)
        ax[0, 2].scatter(x_axis, np.mean(msi_data['north_discharge_current'], axis=1), s=1.0, c=[c_diode])

        ax[1, 2].text(0.5, 0.95, 'diode 4', color='w', transform=ax[1, 2].transAxes)
        ax[1, 2].scatter(x_axis, np.mean(msi_data['north_discharge_voltage'], axis=1), s=1.0, c=[c_diode])
    except:
        pass

    ax[2, 2].text(0.5, 0.95, 'Manget profile', color='w', transform=ax[2, 2].transAxes)
    ax[2, 2].scatter(x_axis, np.mean(msi_data['magnet_profile'], axis=1), s=1.0, c=[c_white])

    ax[2, 3].set_yscale('log')
    ax[2, 3].text(0.5, 0.95, 'Gas pressures (log)', color='w', transform=ax[2, 3].transAxes)
    ax[2, 3].scatter(x_axis, (msi_data['pressure_fill'][:]), s=1.0, c=[c_white], alpha=0.05)
    ax[2, 3].text(0, msi_data['pressure_fill'][0] + 1e-11, f'0')
    for i in range(40):
        ax[2, 3].scatter(x_axis, (msi_data['RGA_partials'][:, i]), s=0.1,
                         c=[mpl.cm.get_cmap('Pastel1')((i % 7) / 6)], alpha=0.09)
        # to avoid text from going off the bottom and ruining tight_layout
        if ax[2, 3].get_ylim()[0] < msi_data['RGA_partials'][i * msi_data.shape[0] // 40, i]:
            ax[2, 3].text(0 + i * msi_data.shape[0] // 40,
                          msi_data['RGA_partials'][i * msi_data.shape[0] // 40, i] + 1e-11, f'{i + 1}')

    plt.tight_layout()
    plt.savefig("summaries/" + name[:-3] + '-time.png')
    plt.close()

    del msi_data
