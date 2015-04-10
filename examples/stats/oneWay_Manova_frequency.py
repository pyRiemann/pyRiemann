import numpy as np
from pylab import *

from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.event import find_events

from pyriemann.permutations import PermutationTestTwoWay,PermutationTest
from pyriemann.estimation import CospCovariances

###############################################################################
## Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1., 3.
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

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True, add_eeg_ref=False,verbose=False)
labels = epochs.events[:, -1] - 2

# get epochs
epochs_data = epochs.get_data()

# compute cospectral covariance matrices
fmin = 2.0
fmax = 40.0
cosp = CospCovariances(window=128,overlap=0.98,fmin=fmin,fmax=fmax,fs = 160.0)
covmats = cosp.fit_transform(epochs_data[:,::4,:]) 

fr = np.fft.fftfreq(128)[0:64]*160
fr = fr[(fr>=fmin) & (fr<=fmax)]

pv = []
Fv = []
# For each frequency bin, estimate the stats
for i in range(covmats.shape[3]):
    p_test = PermutationTest(5000)
    p,F = p_test.test(covmats[:,:,:,i],labels)
    print p_test.summary()
    pv.append(p)
    Fv.append(F[0])
    
plot(fr,Fv,lw=2)
plt.xlabel('Frequency')
plt.ylabel('F-value')

significant = np.array(pv)<0.001
plot(fr,significant,'r',lw=2)
plt.legend(['F-value','p<0.001'])
plt.grid()
plt.show()



