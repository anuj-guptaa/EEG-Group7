#%%
def ():
    import pandas as pd
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scipy.io
    import mne
    import pooch
    from mne import create_info
    from mne.io import RawArray
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf
    import glob
    
    # %%
    # subject = list(range(1,10+1))
    # runs = list(range(1,14+1))
    all_files = []
    for i in range(len(subject)):
        files = eegbci.load_data(subject[i], runs, 'D:\CP-for-DSAI---August-2021\projectCP\datasets\datasets')
        all_files.append(files)
    # %%
    print(len(all_files))
    # all_files
    #%%
    raws = [read_raw_edf(f, preload=True) for f in all_files
    raw_obj = concatenate_raws(raws)
    # %%
    raw_data = raw_obj.get_data()
    # %%
    print("Number of channels: ", str(len(raw_data)))
    print("Number of samples: ", str(len(raw_data[2])))
    # %%
    events, event_ids = mne.events_from_annotations(raw_obj, event_id='auto')
    event_ids
    #%%
    raw_obj
    # %%
    tmin, tmax = -1, 4  # define epochs around events (in s)
    #event_ids = dict(hands=2, feet=3)  # map event IDs to tasks

    epochs = mne.Epochs(raw_obj, events, event_ids, tmin - 0.5, tmax + 0.5, baseline=None, preload=True)
    # %%
    data = epochs._data
    n_events = len(data) # or len(epochs.events)
    print("Number of events: " + str(n_events)) 

    n_channels = len(data[0,:]) # or len(epochs.ch_names)
    print("Number of channels: " + str(n_channels))

    n_times = len(data[0,0,:]) # or len(epochs.times)
    print("Number of time instances: " + str(n_times))
    # %%
    plt.plot(data[14:20,0,:].T)
    plt.title("Exemplar single-trial epoched data, for electrode 0")
    plt.show()

    # %%












    # %%
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    #Define the parameters 
    subject = 1  # use data from subject 1
    runs = [6, 10, 14]  # use only hand and feet motor imagery runs

    #Get data and locate in to given path
    files = eegbci.load_data(subject, runs, '../datasets/')
    #Read raw data files where each file contains a run
    raws = [read_raw_edf(f, preload=True) for f in files]
    #Combine all loaded runs
    raw_obj = concatenate_raws(raws)

    #%%
    










































    # %%
    edf_files
    # %%
    edf_files = glob.glob("D:\CP-for-DSAI---August-2021\projectCP\datasets\datasets\MNE-eegbci-data\\files\eegmmidb\\1.0.0\S001\*.edf")
    # %%
    raws = [read_raw_edf(f, preload=True) for f in edf_files]
    raw_obj = concatenate_raws(raws)
    raw_data = raw_obj.get_data()
    # %%
    raw_obj
    # %%
    print("Number of channels: ", str(len(raw_data)))
    print("Number of samples: ", str(len(raw_data[0])))

    # %%
    raw_data.shape
    # %%
    events, event_ids = mne.events_from_annotations(raw_obj, event_id='auto')
    event_ids
    # %%
    tmin, tmax = -1, 4 
    epochs = mne.Epochs(raw_obj, events, event_ids, tmin - 0.5, tmax + 0.5, baseline=None, preload=True)
    # %%
    data = epochs._data
    # %%
    n_events = len(data) # or len(epochs.events)
    print("Number of events: " + str(n_events)) 

    n_channels = len(data[0,:]) # or len(epochs.ch_names)
    print("Number of channels: " + str(n_channels))

    n_times = len(data[0,0,:]) # or len(epochs.times)
    print("Number of time instances: " + str(n_times))
    # %%
    data.shape
    # %%
    raw_obj.notch_filter([60], filter_length='auto', phase='zero')
    # %%
    # %%
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    #Define the parameters 
    subject = 1  # use data from subject 1
    runs = [3,7,8,9]  # use only hand and feet motor imagery runs

    #Get data and locate in to given path
    files = eegbci.load_data(subject, runs, '../dataset1/')
    #Read raw data files where each file contains a run
    raws = [read_raw_edf(f, preload=True) for f in files]
    #Combine all loaded runs
    raw_obj = concatenate_raws(raws)
    raw_obj
    # %%
    raw_data = raw_obj.get_data()
    print("Number of channels: ", str(len(raw_data)))
    print("Number of samples: ", str(len(raw_data[0])))
    # %%
    events, event_ids = mne.events_from_annotations(raw_obj, event_id='auto')
    event_ids
    # %%
    tmin, tmax = 0, 4
    epochs = mne.Epochs(raw_obj, events, event_ids, tmin, tmax, baseline=None, preload=True)
    # %%
    #Access to the data
    data = epochs._data

    n_events = len(data) # or len(epochs.events)
    print("Number of events: " + str(n_events)) 

    n_channels = len(data[0,:]) # or len(epochs.ch_names)
    print("Number of channels: " + str(n_channels))

    n_times = len(data[0,0,:]) # or len(epochs.times)
    print("Number of time instances: " + str(n_times)) # sample

    # %%
    plt.plot(data[14:20,0,:].T)
    plt.title("Exemplar single-trial epoched data, for electrode 0")
    plt.show()
    # %%
    data.shape
    # %%
    data.shape[0]
    # %%
    events
    # %%
    data.shape
    # %%
    epochs
    # %%
    data[0,0,:2]
    # %%
    raw_obj
    # %%
    raw_data 
    # %%
    raw_data.shape
    # %%

    # %%
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
import mne
import pooch
from mne import create_info
from mne.io import RawArray
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import glob

# %%
subject = list(range(1,10+1))
runs = [3,7,11]
edf_files = glob.glob("D:\CP-for-DSAI---August-2021\projectCP\datasets1\MNE-eegbci-data\\files\eegmmidb\\1.0.0\S001\*.edf")
edf_files
# %%
# C3 = 9 , C4 = 13
electrode_row = [9,13]
# %%
raws = [read_raw_edf(f, preload=True) for f in edf_files]
raw_obj = concatenate_raws(raws)
raw_obj
# %%
raw_data = raw_obj.get_data()
# %%
print("Number of channels: ", str(len(raw_data)))
print("Number of samples: ", str(len(raw_data[2])))
# %%
events, event_ids = mne.events_from_annotations(raw_obj, event_id='auto')
event_ids
#%%
event_ids
# %%
tmin, tmax = 0, 4  
epochs = mne.Epochs(raw_obj, events, event_ids, tmin, tmax, baseline=None, preload=True)
# %%
data = epochs._data
n_events = len(data) # or len(epochs.events)
print("Number of events: " + str(n_events)) 

n_channels = len(data[0,:]) # or len(epochs.ch_names)
print("Number of channels: " + str(n_channels))

n_times = len(data[0,0,:]) # or len(epochs.times)
print("Number of time instances: " + str(n_times))
#%%
num_classes = 3
learning_rate = 0.001
# %%
# data = np.swapaxes(data,1,2)
elc9 = data[:,9,:].reshape(90,1,641)
elc3 = data[:,3,:].reshape(90,1,641)
#%%
elc3.shape
# %%
data_input  = np.concatenate((elc9,elc3), axis=1)
data_input.shape
# %%
X_torch = torch.from_numpy(data_input).float() 
y = epochs.events[:, -1]
y_torch = torch.from_numpy(y)
#%%
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
X_train, X_test, y_train, y_test = train_test_split(X_torch ,y_torch, test_size=0.3)
#%%
from torch.utils.data import TensorDataset

# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader

batch_size = 12

def create_dataloader(X, y, batch_size):
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).long()
    dataset_tensor = TensorDataset(X_tensor, y_tensor)
    dl = torch.utils.data.DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)
    return dl
train_loader = create_dataloader(X_train, y_train, batch_size=batch_size)
test_loader = create_dataloader(X_test, y_test, batch_size=batch_size)
#%%
X_torch = torch.from_numpy(data_input).float() 
y = epochs.events[:, -1]
y_torch = torch.from_numpy(y)
X_torch = torch.swapaxes(X_torch, 0, 1)

# torch.swapaxes(X_torch, 1, 2).shape
X_torch.shape
#%%
X_torch =torch.reshape(X_torch, (X_torch.shape[1],1,X_torch.shape[0],X_torch.shape[2]))
X_torch.shape
# X_torch[,:,:,:].reshape(1,2,90,641)

# %%
# class CNN_Model_v1(nn.Module):
#     def __init__(self):
#         super(CNN_Model_v1, self).__init__()
        
#         # self.conv1 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=4,stride=1)

#         self.L2 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=25,kernel_size=(1,11),stride=1,padding='valid'),
#                                      nn.LeakyReLU(),
#                                      nn.Dropout(p=0.5))
#          self.L3 = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=25,kernel_size=4,stride=4,padding='valid'),
#                                      nn.batch_normalization(),
#                                      nn.LeakyReLU())       

#         # self.L4 = nn.Sequential(nn.MaxPool1d(kernel_size=4,stride=4,padding=1)) #padding 
#         # self.L5 = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=50,kernel_size=4,stride=4,padding='valid'),
#                                     #  nn.LeakyReLU(),
#                                     #  nn.Dropout(p=0.5))
#          # self.L6 = nn.Sequential(nn.MaxPool1d(kernel_size=4,stride=4,padding=1)) #padding        
#         # self.L7 = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=50,kernel_size=4,stride=4,padding='valid'),
#         #                              nn.LeakyReLU(),
#         #                              nn.Dropout(p=0.5))                     
#         #  # self.L8 = nn.Sequential(nn.MaxPool1d(kernel_size=4,stride=4,padding=1)) #padding  
#         # self.L9 = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=50,kernel_size=4,stride=4,padding='valid'),
#         #                              nn.LeakyReLU(),
#         #                              nn.Dropout(p=0.5)) 
       
#         self.L11 = nn.Sequential(nn.Flatten(),nn.Linear(55506, 3))

                      
#     def forward(self, x):
#         print(x.shape)
#         x = self.L2(x)
#         print(x.shape)
#            # x = self.L3(x)
#         # # x = self.L4(x)
#         # print(x.shape)
#         # x = self.L5(x)
#         # # x = self.L6(x)
#         # x = self.L7(x)
#         # # x = self.L8(x)
     
#         x = self.L11(x)
        
#         return x

#%%

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        #using sequential helps bind multiple operations together
        self.layer1 = nn.Sequential(
            #in_channel = 1
            #out_channel = 16
            #padding = (kernel_size - 1) / 2 = 2
            nn.Conv2d(1, 25, kernel_size=(1,11), stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.LeakyReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        #after layer 1 will be of shape [32, 16, 32, 80]
        self.layer2 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(2,1), stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.LeakyReLU())
            # nn.MaxPool2d(kernel_size=3, stride=3))
        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1), stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.LeakyReLU())
            # nn.MaxPool2d(kernel_size=1, stride=3))
        #after layer 2 will be of shape [100, 32, 16, 40]
        self.fc = nn.Linear(222950, 2)
        self.drop_out = nn.Dropout(0.5)  #zeroed 0.2% data
        #after fc will be of shape [100, 10]
        
    def forward(self, x):
        #x shape: [batch, in_channel, img_width, img_height]
        #[32, 1, 64, 161]
        # print("input:",x.shape)
        out = self.layer1(x)
        out = self.drop_out(out)
        # print("after layer 1:",out.shape)
        #after layer 1: shape: [32, 16, 32, 80]
        
        out = self.layer2(out)
        out = self.drop_out(out)
        # print("after layer 2:",out.shape)
        #after layer 2: shape: [100, 32, 16, 40]
        
        out = self.layer3(out)
        out = self.drop_out(out)
        # print("after layer 3:",out.shape)
        
        out = out.reshape(out.size(0), -1)   #can also use .view()
        # print(out.shape)
        #after squeezing: shape: [32, 20480]
        #we squeeze so that it can be inputted into the fc layer
        
        out = self.fc(out)
        print(out.shape)
        #after fc layer: shape: [32, 2]
        return out
#%%
# model = CNN_Model_v1()
# criterion = nn.CrossEntropyLoss()
# learning_rate = 0.01
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%
model = ConvNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# %%

# %%
# num_epochs= 1
# total_step = len(y)
# for epoch in range(num_epochs):
#     for  i in (y_torch):
        
#         #con2d expects (batch, channel, width, height)
        
        
#         #print(images.size())
#         #print(labels.size())
#         # optimizer.zero_grad()
#         # X_torch = classes.long()
#         # outputs = outputs.long()
#         # Forward pass
#         outputs = model(X_torch)
#         print(outputs)
#         loss = criterion(outputs, y_torch)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 100 == 0:
#             sys.stdout.write ('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# %%
num_epochs = 1
train_acc = []
train_losses = []
total_step = len(train_loader)
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        
        #con2d expects (batch, channel, width, height)
        images = images
        labels = labels
        print(images.shape,labels.shape)
        # Forward pass
        outputs = model(images)
        print(outputs.shape)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        total_loss += loss.item()
        correct += (predicted == labels).sum().item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (i+1) % 1 == 0:
            sys.stdout.write ('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    train_acc.append(100 * correct / len(train_loader.dataset))
    train_losses.append(total_loss / len(train_loader))

# %%

(bs, 1, 2, 640)
kernel size = (1,11) , outchannel 25
outputshape = (bs, 25, 2, 630)
conv2 kernel size = (2,1) outchannel 25
