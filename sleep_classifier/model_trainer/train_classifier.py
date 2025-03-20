import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
import joblib
# from sklearn.utils import check_random_state

# for reproducibility:
random_state = 42  


# configuration
EVENT_ID = {"Sleep stage W": 1, "Sleep stage 1/2/3/4": 2, "Sleep stage R": 3}
FREQ_BANDS = {
    "delta": [0.5, 4.5],
    "theta": [4.5, 8.5],
    "alpha": [8.5, 11.5],
    "sigma": [11.5, 15.5],
    "beta": [15.5, 30]
}
NUM_PARTICIPANTS = 10

def load_eeg(participant_id):#, event_id=EVENT_ID):
    """
    Parser for EEG data. Will load EDF with their resp. annotations (for the given sample) and create 30 seconds epochs
    Returns: raw_edf : raw edf with annotations
    """
    ANNOTATION_EVENT_ID = {
        'Sleep stage W': 1,  # Wake (1)
        'Sleep stage 1': 2,  # 3 sleep stages (2)
        'Sleep stage 2': 2,  
        'Sleep stage 3': 2,  
        'Sleep stage R': 3   # Deep sleep (3)
        }
    
    [participant_file] = fetch_data(subjects=[participant_id], recording=[1], path="X:/Brainalytics/sleep_classifier/Data")
    raw_edf = mne.io.read_raw_edf(participant_file[0], stim_channel="Event marker", infer_types=True, preload=True, verbose='error')
    annotation_edf = mne.read_annotations(participant_file[1])
    
    raw_edf.set_annotations(annotation_edf, emit_warning=True)#False)
    events, _ = mne.events_from_annotations(raw_edf, event_id=ANNOTATION_EVENT_ID, chunk_duration=30.0)

    # # max time for epochs
    # tmax = 30.0 - 1.0 / raw_edf.info["sfreq"] # tmax in included
    # # Epochs for classification:
    # epochs = mne.Epochs( # raw signal instace provided with events of 30 secs
    #     raw = raw_edf,
    #     events=events, 
    #     event_id=event_id,
    #     tmin=0.0,
    #     tmax=tmax,
    #     baseline=None,
    #     preload=True,
    #     )
    # return raw_edf, events, epochs

    return mne.Epochs(
        raw_edf, events, EVENT_ID, 0.0, 30.0 - 1/raw_edf.info['sfreq'], 
        baseline=None, preload=True
    ) 

def eeg_power_band(epochs):
    spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds /= np.sum(psds, axis=-1, keepdims=True)
    
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        X.append(psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1))#.reshape(len(psds), -1))
    return np.concatenate(X, axis=1)



def main():
    # load and preprocessing of data
    epochs_list = [load_eeg(p_id) for p_id in range(NUM_PARTICIPANTS)]
    
    # features creation
    X, y, groups = [], [], []
    for group_id, epochs in enumerate(epochs_list):
        X.append(eeg_power_band(epochs))
        y.append(epochs.events[:, 2])
        groups.extend([group_id] * len(epochs.events))
    
    X = np.concatenate(X)
    y = np.concatenate(y)
    groups = np.array(groups)

    # train-test split by participants
    test_ids = [9]  # Last participant as test set, set to 9
    train_mask = ~np.isin(groups, test_ids)
    test_mask = np.isin(groups, test_ids)

    # hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None],
        'max_features': ['sqrt', 'log2']
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state),  # random forest classifier with set rstate
        param_dist,
        n_iter=15,
        cv=LeaveOneGroupOut(),
        n_jobs=-1,
        verbose=2,
        random_state=random_state  
    )
    search.fit(X[train_mask], y[train_mask], groups=groups[train_mask])

    # save overall best model and preprocessing info
    joblib.dump(search.best_estimator_, './models/sleep_classifier.pkl')
    joblib.dump({'freq_bands': FREQ_BANDS, 'event_id': EVENT_ID}, './models/preprocessors/preprocessing_params.pkl') 

    # accuracy score evaluation (got 91%)
    print(f"Test Accuracy: {search.score(X[test_mask], y[test_mask])}")

if __name__ == "__main__":
    main()