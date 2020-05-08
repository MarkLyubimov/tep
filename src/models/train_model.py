import os
import gc
import sys
import time
import logging

import pyreadr as py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class DataTEP(Dataset):

    def __init__(self, X):
        self.X = X
        self.X = self.X.sort_values(['faultNumber', 'simulationRun', 'sample'])
        self.X['index'] = self.X.groupby(['faultNumber', 'simulationRun']).ngroup()
        self.X = self.X.set_index('index')

        self.columns = [
            'xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8',
            'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16',
            'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24',
            'xmeas_25', 'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32',
            'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40',
            'xmeas_41', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9',
            'xmv_10', 'xmv_11']

    def __len__(self):
        return self.X.index.max() + 1

    def __getitem__(self, idx):
        features = self.X.loc[idx][self.columns].values
        target = self.X.loc[idx]['faultNumber'].unique()[0]

        features = torch.tensor(features, dtype=torch.float)[-50:, :]
        target = torch.tensor(target, dtype=torch.long)

        return features, target


class UniModel(torch.nn.Module):
    def __init__(self, NUM_LAYERS, INPUT_SIZE, HIDDEN_SIZE, LINEAR_SIZE, OUTPUT_SIZE, BIDIRECTIONAL, DEVICE):
        super().__init__()

        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.input_size = INPUT_SIZE
        self.linear_size = LINEAR_SIZE
        self.output_size = OUTPUT_SIZE
        self.bidirectional = BIDIRECTIONAL

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.4
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=self.hidden_size * (self.bidirectional + 1), out_features=self.linear_size),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=self.linear_size, out_features=OUTPUT_SIZE),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.head(x[:, -1])

        return x


class TwinModel(torch.nn.Module):
    def __init__(self, NUM_LAYERS, INPUT_SIZE, HIDDEN_SIZE, LINEAR_SIZE, OUTPUT_SIZE, BIDIRECTIONAL, DEVICE):
        super().__init__()

        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.input_size = INPUT_SIZE
        self.linear_size = LINEAR_SIZE
        self.output_size = OUTPUT_SIZE
        self.bidirectional = BIDIRECTIONAL

        self.lstm_1 = nn.LSTM(
            input_size=self.input_size[0],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.4
        )

        self.lstm_2 = nn.LSTM(
            input_size=self.input_size[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.4
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=2 * self.hidden_size * (self.bidirectional + 1), out_features=self.linear_size),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=self.linear_size, out_features=OUTPUT_SIZE),
        )

    def forward(self, x):
        x_1 = x[:, :, :41]
        x_2 = x[:, :, 41:]

        x_1, _ = self.lstm_1(x_1)
        x_2, __ = self.lstm_2(x_2)

        x_3 = torch.cat((x_1[:, -1], x_2[:, -1]), dim=-1)

        x = self.head(x_3)

        return x


def train_loop(model, optimizer, criterion, scheduler, train_dl, val_dl):
    loss_train_all, loss_val_all = [], []
    accuracy_train_all, accuracy_val_all = [], []

    for epoch in range(NUM_EPOCHS):

        start = time.time()
        logger.info(f'EPOCH: {epoch}, LR Rate: {scheduler.get_last_lr()[0]}\n')

        loss_train_epoch, loss_val_epoch = 0, 0
        correct_train_epoch, correct_val_epoch = 0, 0
        n_train, n_val = 0, 0

        model.train()
        for (X_batch_train, y_batch_train) in tqdm(train_dl):
            X_batch_train, y_batch_train = X_batch_train.to(device), y_batch_train.to(device)

            optimizer.zero_grad()
            y_pred_train = model(X_batch_train)
            loss_train = criterion(y_pred_train, y_batch_train)
            loss_train.backward()
            optimizer.step()

            loss_train_epoch += loss_train.item() * y_batch_train.size()[0]
            correct_train_epoch += correct(y_pred_train, y_batch_train)
            n_train += y_batch_train.size()[0]

        scheduler.step()
        model.eval()

        with torch.no_grad():

            for item in model.parameters():
                print(item.grad.mean())

            for (X_batch_val, y_batch_val) in tqdm(val_dl):
                X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)

                y_pred_val = model(X_batch_val)
                loss_val = criterion(y_pred_val, y_batch_val)

                loss_val_epoch += loss_val.item() * y_batch_val.size()[0]
                correct_val_epoch += correct(y_pred_val, y_batch_val)
                n_val += y_batch_val.size()[0]

        loss_mean_train_epoch = loss_train_epoch / n_train
        loss_mean_val_epoch = loss_val_epoch / n_val

        loss_train_all.append(loss_mean_train_epoch)
        loss_val_all.append(loss_mean_val_epoch)

        accuracy_train_epoch = correct_train_epoch / n_train
        accuracy_val_epoch = correct_val_epoch / n_val

        accuracy_train_all.append(accuracy_train_epoch)
        accuracy_val_all.append(accuracy_val_epoch)

        writer.add_scalars('LOSS per epoch', {"train": loss_mean_train_epoch, "val": loss_mean_val_epoch}, epoch)
        writer.add_scalars('ACCURACY per epoch', {"train": accuracy_train_epoch, "val": accuracy_val_epoch}, epoch)

        end = time.time()
        logger.info(f"epoch time: {end - start}")
        logger.info(f"mean loss train: {loss_mean_train_epoch}, mean loss val: {loss_mean_val_epoch}")
        logger.info(f"accuracy train: {accuracy_train_epoch}, accuracy val: {accuracy_val_epoch}")

    return model


def model_eval(model, val_dl):
    model.eval()
    y_ans_val, y_true_val = [], []

    with torch.no_grad():
        for (X_batch_val, y_batch_val) in tqdm(val_dl):
            X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)

            y_pred_val = model(X_batch_val)

            y_pred_prob = F.softmax(y_pred_val.cpu(), dim=-1)
            y_pred_class = y_pred_prob.max(dim=-1)[1]

            y_ans_val += y_pred_class.tolist()
            y_true_val += y_batch_val.tolist()

    return confusion_matrix(y_true_val, y_ans_val, normalize='pred')


def correct(y_pred, target):
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = torch.max(y_pred, dim=1)[1]

    return torch.eq(y_pred, target).sum().item()


if __name__ == "__main__":

    logFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.INFO)
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(logFormatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(streamHandler)
    logger.propagate = False

    logger.info(f'Script started')

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    logger.info(f'Selected GPU: {device}')

    # reading data in .R format
    logger.info("reading data")
    a1 = py.read_r("../../data/raw/dataverse_files/TEP_FaultFree_Training.RData")
    a2 = py.read_r("../../data/raw/dataverse_files/TEP_Faulty_Training.RData")
    a3 = py.read_r("../../data/raw/dataverse_files/TEP_FaultFree_Testing.RData")
    a4 = py.read_r("../../data/raw/dataverse_files/TEP_Faulty_Testing.RData")

    # concatenating the train and the test dataset
    raw_train = pd.concat([a1['fault_free_training'], a2['faulty_training']])
    raw_test = pd.concat([a3['fault_free_testing'], a4['faulty_testing']])

    logger.info(f'raw_train size: {len(raw_train)}, raw_test size: {len(raw_test)}')

    raw_train['index'] = raw_train['faultNumber'] * 500 + raw_train['simulationRun'] - 1
    raw_test['index'] = raw_test['faultNumber'] * 500 + raw_test['simulationRun'] - 1

    features = [
        'xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8',
        'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16',
        'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24',
        'xmeas_25', 'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32',
        'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40',
        'xmeas_41', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9',
        'xmv_10', 'xmv_11'
    ]

    logger.info("train-test splitting")

    simulation_idx = raw_train[['index', 'faultNumber']].drop_duplicates()

    X_train_idx, X_val_idx = train_test_split(simulation_idx['index'],
                                              stratify=simulation_idx['faultNumber'],
                                              test_size=0.2,
                                              random_state=42)

    X_train = raw_train[raw_train['index'].isin(X_train_idx)].drop('index', axis=1)
    X_val = raw_train[raw_train['index'].isin(X_val_idx)].drop('index', axis=1)
    X_test = raw_test.drop('index', axis=1)

    logger.info("data scaling: training")
    scaler = StandardScaler()
    scaler.fit(X_train[features])

    logger.info("data scaling: transforming")
    X_train[features] = scaler.transform(X_train[features])
    X_val[features] = scaler.transform(X_val[features])
    X_test[features] = scaler.transform(X_test[features])

    logger.info("preparing datasets and dataloaders")
    BATCH_SIZE = 64
    NUM_CLASSES = 21

    train_ds = DataTEP(X_train)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_ds = DataTEP(X_val)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 4)

    test_ds = DataTEP(X_test)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE * 4)

    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001

    NUM_LAYERS = 2
    HIDDEN_SIZE = 256
    LINEAR_SIZE = 128
    BIDIRECTIONAL = True

    model = UniModel(NUM_LAYERS=NUM_LAYERS, INPUT_SIZE=52, HIDDEN_SIZE=HIDDEN_SIZE,
                     LINEAR_SIZE=LINEAR_SIZE, OUTPUT_SIZE=NUM_CLASSES, BIDIRECTIONAL=BIDIRECTIONAL, DEVICE=device)

    # model = TwinModel(NUM_LAYERS=NUM_LAYERS, INPUT_SIZE=[41, 11], HIDDEN_SIZE=HIDDEN_SIZE,
    #                  LINEAR_SIZE=LINEAR_SIZE, OUTPUT_SIZE=NUM_CLASSES, BIDIRECTIONAL=BIDIRECTIONAL, DEVICE=device)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    torch.manual_seed(42)

    writer = SummaryWriter(comment=f'NL{NUM_LAYERS}_H{HIDDEN_SIZE}_L{LINEAR_SIZE}_B{BIDIRECTIONAL}')

    logger.info("model training: started")
    model = train_loop(model=model, optimizer=optimizer, criterion=criterion,
                       scheduler=scheduler, train_dl=train_dl, val_dl=val_dl)

    logger.info("model training: finished")
    fdr = model_eval(model=model, val_dl=val_dl)
