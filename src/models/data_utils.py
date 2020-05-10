import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class DataTEP(Dataset):

    def __init__(self, X, s_list):
        self.X = X
        self.X = self.X.sort_values(['faultNumber', 'simulationRun', 'sample'])
        self.X['index'] = self.X.groupby(['faultNumber', 'simulationRun']).ngroup()
        self.X = self.X.set_index('index')

        self.s_list = s_list
        self.l_list = [100]

        self.features = [
            'xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9',
            'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16', 'xmeas_17',
            'xmeas_18', 'xmeas_19', 'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25',
            'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32', 'xmeas_33',
            'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40', 'xmeas_41',
            'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11'
        ]

    def __len__(self):
        return self.X.index.nunique() * len(self.s_list) * len(self.l_list)

    def __getitem__(self, idx):
        fault_sim_idx = idx // (len(self.s_list) * len(self.l_list))

        start_length_idxs = idx % (len(self.s_list) * len(self.l_list))

        start_idx = self.s_list[start_length_idxs // len(self.l_list)]
        seq_length = self.l_list[start_length_idxs % len(self.l_list)]

        #         print(start_idx, start_idx+seq_length)

        features = self.X.loc[fault_sim_idx][self.features].values[start_idx: (start_idx + seq_length), :]
        target = self.X.loc[fault_sim_idx]['faultNumber'].unique()[0]

        features = torch.tensor(features, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.long)

        return features, target


def collate_fn(batch):
    sequences = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    lengths = torch.LongTensor([len(x) for x in sequences])
    lengths, idx = lengths.sort(0, descending=True)

    sequences = [sequences[i] for i in idx]

    labels = torch.tensor(labels, dtype=torch.long)[idx]

    sequences_padded = pad_sequence(sequences, batch_first=True)

    return sequences_padded, lengths, labels