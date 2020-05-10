from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class UniModel(nn.Module):
    def __init__(self, NUM_LAYERS, INPUT_SIZE, HIDDEN_SIZE, LINEAR_SIZE, OUTPUT_SIZE, BIDIRECTIONAL):
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

    def forward(self, x, x_length):
        x_packed = pack_padded_sequence(x, x_length, batch_first=True)

        x_lstm_out, _ = self.lstm(x_packed)

        x_unpacked, _ = pad_packed_sequence(x_lstm_out, batch_first=True)

        x = self.head(x_unpacked[:, -1])

        return x