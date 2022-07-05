import torch
import time
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import *



def train_evaluate(
    model,              # model's instance
    optimizer,          # optimizer instance
    criterion,          # loss function
    dataloaders,        # dictionary of train and (optional) valid dataloaders
    epochs=10,          # number of epochs
    # lr=3e-4,          # learning rate (default: Karpathy’s constant)
    device='auto',      # device for computations
    score_funcs={},     # score functions to use from sklearn
    checkpoint='',      # name of the file for the checkpoint
    save_best=False
    ):

    # place the model on the selected device
    if device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # create a dictionary dataloader if a single dataloader is passed
    if not isinstance(dataloaders, dict):
        dataloaders = {'train': dataloaders}
    
    # set up a dictionary of items to track
    track_list = ['epoch', 'time']
    # check number of phases to track
    if len(dataloaders) > 1:
        phases = ['train', 'valid']
    else:
        phases = ['train']
    # add loss tracking for each phase
    track_list.extend([phase + '_loss' for phase in phases])
    # add score functions tracking for each phase
    if score_funcs:
        key_list = [key for key in score_funcs.keys()]
        track_list.extend(
            [phase + '_' + key for key in key_list for phase in phases]
            )
    # instantiate the dictionary log to track
    train_log = {x: [] for x in track_list}
    # print(train_log)

    # initialize time for logging
    train_time = float()
    best_loss = float()

    # iterate through epochs
    for epoch in tqdm(range(1, epochs+1)):

        # set up
        start_time = time.time()
        train_epoch_loss, valid_epoch_loss = [], []

        # iterate through phases:
        for phase in phases:

            # set up
            y_true, y_pred = [], []

            # set model to train or eval mode
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()

            # iterate over batches
            for inputs, labels in tqdm(dataloaders[phase], leave=False):

                # place the data on the selected device
                inputs, labels = inputs.to(device), labels.to(device)

                # enable autograd differentiation only for training
                with torch.set_grad_enabled(phase == 'train'):

                    # forward propagation
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward propagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    # logging the loss
                    if phase == 'train':
                        train_epoch_loss.append(loss.item())
                    if phase == 'valid':
                        valid_epoch_loss.append(loss.item())

                # score functions calculation
                if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
                    # moving labels & outputs to CPU arrays
                    labels = labels.detach().cpu().numpy()
                    outputs = outputs.detach().cpu().numpy()
                    # save the labels & outputs for later
                    y_true.extend(labels.tolist())
                    y_pred.extend(outputs.tolist())
            
            y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
            if len(y_pred.shape) == 2 and y_pred.shape[1] > 1: #We have a classification problem, convert to labels
                y_pred = np.argmax(y_pred, axis=1)
            #Else, we assume we are working on a regression problem
        
            # calculate and logging the score functions
            for key, score_func in score_funcs.items():
                key = phase + '_' + key
                try:
                    train_log[key].append(score_func(y_true, y_pred))
                except:
                    train_log[key].append(float("NaN"))

        # stop timer and check time
        end_time = time.time()
        epoch_time = end_time - start_time
        train_time += epoch_time

        # logging the epochs, time, losses
        for key in train_log.keys():
            if key == 'epoch':
                train_log[key].append(epoch)
            elif key == 'time':
                train_log[key].append(round(train_time, 2))
            elif key == 'train_loss':
                train_log[key].append(np.mean(train_epoch_loss))
            elif key == 'valid_loss':
                train_log['valid_loss'].append(
                    np.mean(valid_epoch_loss) if not valid_epoch_loss else 'nan'
                    )
    
        # save a checkpoint
        if checkpoint:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'log' : train_log
                },
                checkpoint
            )
        
        # deep copy the model
        if save_best and phase == 'valid' and train_log['valid_loss'][-1] < best_loss:
            best_loss = train_log['valid_loss'][-1]
            best_model = copy.deepcopy(model.state_dict())

    # return a dataframe object with all the logging information
    if save_best:
        return (
            pd.DataFrame.from_dict(train_log).set_index('epoch'),
            model.load_state_dict(best_model)
        )
    else:
        return pd.DataFrame.from_dict(train_log).set_index('epoch')