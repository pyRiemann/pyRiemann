#generic import
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

#mne import
from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.event import find_events
from mne.decoding import CSP

#pyriemann import
sys.path.append('../../')
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances

#sklearn imports
from sklearn.cross_validation import cross_val_score, LeaveOneLabelOut
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA

###############################################################################
## Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1., 2.
event_id = dict(hands=2, feet=3)
subjects = range(1,109)
# There is subject where MNE can read the file
subject_to_remove = [88,89,92,100]

for s in subject_to_remove:
    if s in subjects:
        subjects.remove(s)

runs = [6, 10, 14]  # motor imagery: hands vs feet


classifiers = {
                'mdm'  : make_pipeline(Covariances(),MDM(metric='riemann')),
                'tsLR' : make_pipeline(Covariances(),TangentSpace(),LogisticRegression()),
                'csp'  : make_pipeline(CSP(n_components=4, reg=None, log=True),LDA())       
                }

# cross validation

results = np.zeros((len(subjects),len(classifiers)))

for s,subject in enumerate(subjects):
    
    print('Processing Subject %s' %(subject))
    raw_files = [read_raw_edf(f, preload=True,verbose=False) for f in eegbci.load_data(subject, runs)]
    raw = concatenate_raws(raw_files)

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    # subsample elecs
    picks = picks[::2]
    
    # Apply band-pass filter
    raw.filter(7., 35., method='iir',picks=picks)
    
    events = find_events(raw, shortest_event=0, stim_channel='STI 014',verbose=False)



    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True, add_eeg_ref=False,verbose = False)
                    
    labels = epochs.events[:, -1] - 2
    X = epochs.get_data()
    
    for i,c in enumerate(classifiers):
        
        r = cross_val_score(classifiers[c],X,labels,cv=10,n_jobs=-1)
        results[s,i] = np.mean(r)

df = pd.DataFrame(results,index=subjects,columns=classifiers.keys())

print df
df.to_csv('./results/single_subject_comparison.csv')

df.boxplot()
plt.show()