from tqdm import tqdm
import mat73
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold,GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay,precision_score,recall_score,confusion_matrix
from imblearn.over_sampling import SMOTE

from sklearn.svm import SVC
from sklearn.decomposition import PCA

import joblib

## Import MATLAB Data
#processing MATLAB Function

import mat73
import numpy as np
import mne


chan_names = ['EOGv', 'Fp1', 'Fp2', 'AF3', 'AF4', 'EOGh', 'F7', 'F5', 'F3', 'F1' , 'Fz' , 'F2' , 'F4' , 'F6' , 'F8' , 'FT7' , 'FC5' , 'FC3' ,
              'FC1' , 'FCz' , 'FC2' , 'FC4' , 'FC6' , 'FT8' , 'T7' , 'C5' , 'C3' , 'C1' , 'Cz' , 'C2' , 'C4' , 'C6' , 'T8' , 'TP7' , 'CP5' ,
              'CP3' , 'CP1' , 'CPz' , 'CP2' , 'CP4' , 'CP6' , 'TP8' , 'P9' , 'P7' , 'P5' , 'P3' , 'P1' , 'Pz' , 'P2' , 'P4' , 'P6' , 'P8' ,
            'P10' , 'PO7' , 'PO3' , 'POz' , 'PO4' , 'PO8' , 'O1' , 'Oz' , 'O2' , 'EMGf' , 'lead_gas','lead_brake','dist_to_lead',
              'wheel_X','wheel_Y','gas','brake']



def load_MATLAB_data(dir=None, participant_data=None):
    if dir==None and participant_data==None: raise Exception("Must provide directory or dicitionary")
    # print(dir)
    #load data
    if participant_data == None:
      participant_data = mat73.loadmat(dir)

    eeg_names = chan_names[1:61]
    eeg_names.pop(4)


    #create MNE info object
    sfreq = 200
    n_channels = 59
    # Initialize an info structure
    info = mne.create_info(
            ch_names = eeg_names,
            ch_types = ['eeg']*n_channels,
            sfreq    = sfreq
            )

    info.set_montage('standard_1020')

    #create labels
    Y  = participant_data['mrk']['y']
    Y = np.rollaxis(Y, 1, 0)
    _, ind = np.where(Y>0)
    np.unique(ind)
    labels = ind + 1

    #create event dict
    event_id = dict(car_normal = 1, car_brake = 2,	car_hold = 3,	car_collision = 4,	react_emg = 5)
    eventLength = Y.shape[0]
    ev = np.array([int(participant_data['mrk']['time'][i]/5) for i in range(eventLength)])
    #delete duplicates
    # ev = np.delete(ev, 578, 0)
    # labels = np.delete(labels, 578, 0)

    events = np.column_stack((np.array(ev),
                          np.zeros(eventLength,  dtype = int),
                          np.array(labels)))

    #get time intervals around each y
    # stim_slices = [participant_data['cnt']['x'][int(idx/5)-340:int(idx/5)+240] for idx in participant_data['mrk']['time']]
    stim_slices = [participant_data['cnt']['x'][int(idx/5)-60:int(idx/5)+240] for idx in participant_data['mrk']['time']]


    # stim_slices.pop(578) #remove duplicates

    #reshape data
    npdata = np.array(stim_slices, dtype=object,)
    npdata = np.swapaxes(npdata, 1,2)
    npdata = np.delete(npdata, 0, 1)
    npdata = np.delete(npdata, 4, 1)
    npdata = np.delete(npdata, np.s_[59:], 1)

    # tmin = 0
    #     # Create the :class:`mne.EpochsArray` object
    # epochs = mne.EpochsArray(npdata, info, events, tmin, event_id)

    raw_data = np.swapaxes(np.array(participant_data['cnt']['x']), 0, 1)

    raw_data = np.delete(raw_data, 0, 0)
    raw_data = np.delete(raw_data, 4, 0)
    raw_data = np.delete(raw_data, np.s_[59:], 0)

    raw_eeg = mne.io.RawArray(raw_data,info, verbose=True)


    # epochs = mne.Epochs(raw_eeg, events, event_id=event_id, tmin=-1.7, tmax=1.2, preload=True, event_repeated='drop', verbose=True)
    epochs = mne.Epochs(raw_eeg, events, event_id=event_id, tmin=-0.3, tmax=1.2, preload=True, event_repeated='drop', verbose=True)



    return raw_eeg, epochs


## Develop ML Model
def get_filtered_data(epoch, vpae_dict):
    non_targets = []
    data_pairs = list(zip(vpae_dict['mrk']['time'], vpae_dict['mrk']['time'][1:]))
    data_pairs = [(int(a[0]/5), int(a[1]/5)) for a in data_pairs]
    # npTarget = np.delete(npTarget, npFiltered, 1)
    for pr in data_pairs:
        counter = pr[0] + 600
        while (counter < pr[1]-900):
            non_targets.append(vpae_dict['cnt']['x'][counter:counter+301])
            counter += 301

    # data_pairs = [(vpae_dict['mrk']['time'][i], vpae_dict['mrk']['time'][i+1]) for i in range(len(vpae_dict['mrk']['time'])-1)]
    # for vpae_dict in mat_dicts:
    # npFiltered = np.where(np.std(epoch['car_normal'].average().get_data(), axis=1) > 2)
    # print(npFiltered)

    npNonTarget = np.array(non_targets)
    npNonTarget = np.swapaxes(npNonTarget, 1,2)
    npNonTarget = np.delete(npNonTarget, 0, 1)
    npNonTarget = np.delete(npNonTarget, 4, 1)
    npNonTarget = np.delete(npNonTarget, np.s_[59:], 1)

    return npNonTarget#, npFiltered

def get_target_based_on_time(vpae_dict, length, npFiltered):

    indices = list(np.where(vpae_dict['mrk']['y'][1] == 1.0)[0])
    times = [vpae_dict['mrk']['time'][x] for x in indices]
    targets = np.array([vpae_dict['cnt']['x'][int(idx/5)+length-1:int(idx/5)+length+300] for idx in times])
        # targets = [vpae_dict['cnt']['x'][int(idx/5)+len:int(idx/5)+len+1] for idx in times]


    npTarget = np.array(targets)
    npTarget = np.swapaxes(npTarget, 1,2)
    npTarget = np.delete(npTarget, 0, 1)
    npTarget = np.delete(npTarget, 4, 1)
    npTarget = np.delete(npTarget, np.s_[59:], 1)
    # npTarget = np.delete(npTarget,npFiltered, 1)

    # npNonTarget = npNonTarget - np.mean(npNonTarget)
    # npNonTarget = npNonTarget / np.std(npNonTarget)
    # npTarget = npTarget - np.mean(npTarget)
    # npTarget = npTarget / np.std(npTarget)

    return npTarget

def create_dataset(npNonTarget, npTarget):
    TR = npTarget 
    TR = np.swapaxes(TR, 1, 2)
    TR = np.swapaxes(TR, 0, 1)
    TR = np.mean(TR, axis=2)
    TR = np.swapaxes(TR, 0, 1)

    FL = npNonTarget 
    FL = np.swapaxes(FL, 1, 2)
    FL = np.swapaxes(FL, 0, 1)
    FL = np.mean(FL, axis=2)
    FL = np.swapaxes(FL, 0, 1)


    # TR = sc.fit_transform(TR)
    # FL = sc.fit_transform(FL)

    TrueLabels = [1 for sample in range(TR.shape[0])]
    FalseLabels = [0 for sample in range(FL.shape[0])]

    X = np.concatenate((TR, FL))
    y = np.concatenate((TrueLabels, FalseLabels))


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test
    
def create_PCA_LDA_OVER(X_train, X_test, y_train, y_test):
    pca = PCA(n_components=10)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    oversample = SMOTE()
    Xs_train, Xs_test, ys_train, ys_test = X_train, X_test, y_train, y_test
    Xs_train, ys_train = oversample.fit_resample(Xs_train, ys_train)

    pca_os = PCA(n_components=10)
    Xs_train = pca_os.fit_transform(Xs_train)
    Xs_test = pca_os.transform(Xs_test)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    grid = dict()
    grid['solver'] = ['eigen','lsqr']
    grid['shrinkage'] = ['auto',0.2,1,0.3,0.5]
    search = GridSearchCV(LDA(), grid, scoring='precision', cv=cv, n_jobs=8)
    results = search.fit(Xs_train, ys_train)


    LDA_final=LDA(shrinkage='auto', solver='eigen')
    LDA_final.fit(Xs_train,ys_train)
    ys_pred = LDA_final.predict(Xs_test)


    ConfusionMatrixDisplay.from_predictions(ys_test, ys_pred)
    print(X_test.shape)

    acc = accuracy_score(ys_test, ys_pred)
    ys_pred_proba = LDA_final.predict_proba(Xs_test)[:, 1]
    auc_score = roc_auc_score(ys_test, ys_pred_proba)
    f1 = f1_score(ys_test, ys_pred)

    # save the model to disk
    filename = 'PCA_LDA_OVER_model.sav'
    joblib.dump(LDA_final, filename)

    return acc, auc_score, f1

def train_one_trial(dir, length):
    
    participant_data = mat73.loadmat(dir)
    raw, epochs = load_MATLAB_data(participant_data=participant_data)
    npNonTarget = get_filtered_data(epochs, participant_data)
    npTarget = get_target_based_on_time(participant_data, length, _)

    X_train, X_test, y_train, y_test = create_dataset(npNonTarget, npTarget)

    acc, auc_score, f1 = create_PCA_LDA_OVER(X_train, X_test, y_train, y_test)

    return acc, auc_score, f1

if __name__ == "__main__":
    acc, auc_score, f1 = train_one_trial(dir="VPae.mat", length=100)
