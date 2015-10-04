import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.event import find_events

from pyriemann.stats import PermutationTestTwoWay
from pyriemann.estimation import Covariances

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

# Apply band-pass filter
raw.filter(7., 35., method='iir')

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

# compute covariance matrices
covmats = Covariances().fit_transform(epochs_data) 

session = np.array([1,2,3]).repeat(15)

p_test = PermutationTestTwoWay(5000)
p,F = p_test.test(covmats,session,labels,['session','handsVsFeets'])
p_test.plot()
print p_test.summary()

plt.show()


