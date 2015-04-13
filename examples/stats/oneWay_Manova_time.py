import numpy as np
from pylab import *

from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.event import find_events

from pyriemann.permutations import PermutationTestTwoWay,PermutationTest
from pyriemann.estimation import Covariances

###############################################################################
## Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -2., 6.
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_files = [read_raw_edf(f, preload=True,verbose=False) for f in eegbci.load_data(subject, runs) ]
raw = concatenate_raws(raw_files)

# strip channel names
raw.info['ch_names'] = [chn.strip('.') for chn in raw.info['ch_names']]



events = find_events(raw, shortest_event=0, stim_channel='STI 014')
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

raw.filter(7., 35., method='iir',picks=picks)


epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True, add_eeg_ref=False,verbose=False)
labels = epochs.events[:, -1] - 2

# get epochs
epochs_data = epochs.get_data()

# compute cospectral covariance matrices

covest = Covariances()

Fs = 160
window = 2*Fs
Nwindow = 20
Ns = epochs_data.shape[2]
step = int((Ns-window)/Nwindow )
time_bins = range(0,Ns-window,step)

pv = []
Fv = []
# For each frequency bin, estimate the stats
for t in time_bins:
    covmats = covest.fit_transform(epochs_data[:,::1,t:(t+window)])
    p_test = PermutationTest(5000)
    p,F = p_test.test(covmats,labels)
    print p_test.summary()
    pv.append(p)
    Fv.append(F[0])

time = np.array(time_bins)/float(Fs) + tmin
plot(time,Fv,lw=2)
plt.xlabel('Time')
plt.ylabel('F-value')

significant = np.array(pv)<0.001
plot(time,significant,'r',lw=2)
plt.legend(['F-value','p<0.001'])
plt.grid()
plt.show()



