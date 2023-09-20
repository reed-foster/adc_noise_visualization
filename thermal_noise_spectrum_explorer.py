import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import glob

scale_factor = (2**16)*(10**(46/20))
fsamp = 4096e6
beta = 38
# spectral width of kaiser window
width = int(np.ceil(2*(1+(beta/np.pi)**2)**0.5))
fsamp = fsamp/(4096//50)

all_files = list(sorted(glob.glob('biastini_adc_raw_data_50ohm_thermalnoise/*.mat')))

digits = lambda N_max: int(np.ceil(np.log10(N_max)))

def file_select():
    file_list = []
    d = input(f'select one or more files to view (e.g. 1,2 or 1-3,7): ')
    runs = d.split(',')
    for run in runs:
        if '-' in run:
            lohi = run.split('-')
            low = int(lohi[0])
            high = int(lohi[1])
            if low >= high:
                return None
            if low < 0 or high >= len(all_files):
                return None
            file_list += list(range(low, high+1))
        else:
            d = int(run)
            if d >= len(all_files) or d < 0:
                return None
            file_list.append(d)
    return file_list

def generate_histogram(file_list, merge):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    alpha = 0.5
    colors_tup = [(int(c[1:3],16)/255, int(c[3:5],16)/255, int(c[5:7],16)/255, alpha) for c in colors]
    plt.figure()
    # if merge, then combine all trials from all files into one big array and then make one or two histograms (separated based on shielding if all files have two channels)
    if merge:
        n_channels = -1
        merged_rms = np.zeros((1,1))
        for n,file in enumerate(file_list):
            mdata = scipy.io.loadmat(all_files[file])
            rms = mdata['rms_noise_16b'].T / scale_factor * 1e6
            if n == 0:
                n_channels = rms.shape[1]
                merged_rms = rms.copy()
            else:
                if rms.shape[1] != n_channels:
                    n_channels = 1
                    merged_rms.reshape(merged_rms.size)
                if n_channels == 1:
                    # resize array and add new elements
                    merged_rms.resize(merged_rms.shape[0]+rms.size)
                    merged_rms[-rms.size:] = rms.flatten()
                else:
                    merged_rms.resize((merged_rms.shape[0]+rms.shape[0],n_channels))
                    merged_rms[-rms.shape[0]:,:] = rms[:,:]
        cindex = 0
        bins = max(merged_rms.shape[0]//10, 50)
        bins = np.histogram(merged_rms, bins=bins)[1]
        if merged_rms.shape[1] > 1:
            # more than 1 channel
            for i in range(2):
                plt.hist(merged_rms[:,i], label=mdata['shielding'][i], bins=bins, fc=colors_tup[cindex % len(colors_tup)])
                cindex += 1
            plt.legend()
        else:
            # only 1 channel
            plt.hist(merged_rms, bins=bins, fc=colors_tup[cindex % len(colors_tup)])
            cindex += 1
        plt.title(f'merged data for files {file_list}')
    else:
        cindex = 0
        for file in file_list:
            mdata = scipy.io.loadmat(all_files[file])
            rms = mdata['rms_noise_16b'] / scale_factor * 1e6
            bins = max(rms.shape[-1]//10, 50)
            bins = np.histogram(rms, bins=bins)[1]
            label_prefix = all_files[file].split('/')[-1].split('_')[-1].split('.')[0]
            if rms.shape[0] > 1:
                # more than 1 channel
                for i in range(2):
                    plt.hist(rms[i,:], label=label_prefix + ' ' + mdata['shielding'][i], bins=bins, fc=colors_tup[cindex % len(colors_tup)])
                    cindex += 1
            else:
                # only 1 channel
                plt.hist(rms, label=label_prefix, bins=bins, fc=colors_tup[cindex % len(colors_tup)])
                cindex += 1
        plt.legend(ncol=2 if len(file_list) > 5 else 1)
    plt.yscale('log')
    plt.xlabel('input-referred rms noise voltage [uV]')
    plt.ylabel('counts')
    plt.show()

def plot_timeseries(file, channel, trial):
    mdata = scipy.io.loadmat(all_files[file])
    n_channels = mdata['rms_noise_16b'].shape[0]
    n_trials = mdata['rms_noise_16b'].shape[1]
    v_t = mdata['tdata_16b'][channel,trial,:] if n_channels > 1 else mdata['tdata_16b'][trial,:]
    v_t /= scale_factor # refer to input
    t = np.linspace(0, (len(v_t)-1)/fsamp, len(v_t))
    fig, ax = plt.subplots(1,4)
    ax[0].plot(t*1e6, v_t*1e6)
    ax[0].set_xlabel('t [us]')
    ax[0].set_ylabel('V [uV]')
    # calculate periodogram
    [f, Pxx] = scipy.signal.periodogram(v_t, fsamp, window=('kaiser', beta), axis=0, scaling='density')
    ax[1].semilogy(f/1e6, Pxx)
    ax[1].set_xlabel('f [MHz]')
    ax[1].set_ylabel('PSD [V**2/Hz]')
    ax[2].loglog(f, Pxx)
    ax[2].set_xlabel('f [Hz]')
    ax[2].set_ylabel('PSD [V**2/Hz]')
    ax[1].set_ylim([1e-20, 1e-12])
    ax[2].set_ylim([1e-20, 1e-12])
    if n_channels > 1:
        ax[0].set_title(f'time-series {mdata["shielding"][channel]}')
        ax[1].set_title(f'linear-frequency PSD {mdata["shielding"][channel]}')
        ax[2].set_title(f'log-frequency PSD {mdata["shielding"][channel]}')
    else:
        ax[0].set_title('time-series')
        ax[1].set_title('linear-frequency PSD')
        ax[2].set_title('log-frequency PSD')
    # spectrogram
    [f,t,Sxx] = scipy.signal.spectrogram(v_t, fsamp, window=('kaiser', beta), axis=0, nperseg=512, noverlap=128, scaling='density', mode='psd')
    pcm = ax[3].pcolormesh(t*1e6,f/1e6,Sxx,norm=clrs.LogNorm(vmin=Sxx.min(),vmax=Sxx.max()))
    cb = fig.colorbar(pcm, ax=ax[3])
    cb.set_label('PSD [V**2/Hz]')
    ax[3].set_title(f'spectrogram {mdata["shielding"][channel]}')
    ax[3].set_ylabel('f [MHz]')
    ax[3].set_xlabel('t [us]')
    plt.show()

# TUI
file_list = []
while True:
    if len(file_list) == 0:
        print('no file selected')
    else:
        print(f'selected files {file_list}')
    d = input(f'enter L to list all files, S to change file selection, H to generate a histogram, D to view time-series and spectral data, R to refresh the file list, or Q to quit: ')
    if d == 'q' or d == 'Q':
        break
    elif d == 'r' or d == 'R':
        all_files = list(sorted(glob.glob('biastini_adc_raw_data_50ohm_thermalnoise/*.mat')))
    elif d == 'L' or d == 'l':
        for n,f in enumerate(all_files):
            print('({n:0{width}}): {f}'.format(n=n,width=digits(len(all_files)),f=f.split('/')[-1]))
    elif d == 's' or d == 'S':
        file_list = file_select()
        if file_list is None:
            print('invalid file selection')
            file_list = []
    elif d == 'h' or d == 'H':
        generate_histogram(file_list, False)
    elif d == 'hm' or d == 'HM':
        generate_histogram(file_list, True)
    elif d == 'd' or d == 'D':
        if len(file_list) > 1:
            print('cannot plot time-series data for more than 1 file simultaneously, please select just one file')
        else:
            mdata = scipy.io.loadmat(all_files[file_list[0]])
            if 'tdata_16b' not in mdata.keys():
                print('file does not have time-series data in it, please select a different file')
            else:
                n_channels = mdata['rms_noise_16b'].shape[0]
                n_trials = mdata['rms_noise_16b'].shape[1]
                while True:
                    d = input(f'select a channel {range(n_channels)} and trial (0-{n_trials-1}) [ch,tr], or press T to select trials with noise voltage above a threshold: ')
                    if 't' in d or 'T' in d:
                        if len(d) == 1:
                            channels = range(n_channels)
                        else:
                            channels = [int(d[1:])]
                        # list trials with rms noise above threshold
                        t = input(f'threshold (uV): ')
                        thresh = int(t)
                        for channel in channels:
                            for trial in range(n_trials):
                                rms_uV = mdata['rms_noise_16b'][channel,trial] / scale_factor * 1e6 
                                if rms_uV >= thresh:
                                    print('({ch:0{w_ch}},{tr:0{w_tr}}): {rms_uV}'.format(ch=channel,w_ch=digits(n_channels),tr=trial,w_tr=digits(n_trials),rms_uV=rms_uV))
                    if ',' in d:
                        digits = d.split(',')
                        if digits[0].isdigit() and int(digits[0]) >= 0 and int(digits[0]) <= n_channels - 1:
                            if digits[1].isdigit() and int(digits[1]) >= 0 and int(digits[1]) <= n_trials - 1:
                                channel = int(digits[0])
                                trial = int(digits[1])
                                plot_timeseries(file_list[0], channel, trial)
                        break
                    if d == 'q' or d == 'Q':
                        break
                else:
                    print('invalid input')
    else:
        print('invalid input')

