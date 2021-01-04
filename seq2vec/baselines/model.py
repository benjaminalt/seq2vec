import os

import torch


class Conv1DNet(torch.nn.Module):
    """
    Reduces a batch of time series of shape (batch_size, seq_len, x) to a batch of embeddings (batch_size, y).
    """

    class ResidualConv1DCell(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)

        def forward(self, x):
            conv1 = torch.nn.SELU()(self.conv1(x))
            conv2 = torch.nn.SELU()(self.conv2(x))
            return conv1 + conv2 + x

    def __init__(self, batch_size, seq_len, input_size, output_size=32):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.input_limits = None
        self.additional_input_limits = None
        self.output_limits = None

        # Idea: Successively reduce the sequence length by 2
        modules = []
        current_seq_len = self.seq_len
        out_channels = 8
        in_channels = self.input_size
        while current_seq_len > 1:
            modules.append(Conv1DNet.ResidualConv1DCell(in_channels, out_channels))
            modules.append(torch.nn.MaxPool1d(2))
            in_channels = out_channels
            out_channels = out_channels * 2 if out_channels < 64 else out_channels
            current_seq_len = int(current_seq_len / 2)
        self.hidden_layers = torch.nn.Sequential(*modules)
        self.output_layer = torch.nn.Linear(in_channels, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (batch_size, seq_len, input_size) -> (batch_size, input_size, seq_len)
        x = self.hidden_layers(x)
        x = x.squeeze()                                 # Remove now redundant time dimension
        assert len(x.size()) == 2
        return torch.nn.SELU()(self.output_layer(x))

    def save(self, archive_dir, filename=None):
        print("Saving model to {}".format(archive_dir))
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        filename = filename if filename is not None else "Conv1DNet.pt"
        torch.save({
            "state_dict": self.state_dict(),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "input_size" : self.input_size,
            "output_size" : self.output_size,
            "input_limits": self.input_limits,
            "additional_input_limits": self.additional_input_limits,
            "output_limits": self.output_limits
        }, os.path.join(archive_dir, filename))

    @staticmethod
    def load(archive_file, device):
        print("Loading model from {}".format(archive_file))
        checkpoint = torch.load(archive_file, map_location=device)
        model = Conv1DNet(checkpoint["batch_size"], checkpoint["seq_len"], checkpoint["input_size"],
                          checkpoint["output_size"])
        model.load_state_dict(checkpoint["state_dict"])
        input_limits = checkpoint["input_limits"].to(device)
        additional_input_limits = checkpoint["additional_input_limits"].to(device)
        output_limits = checkpoint["output_limits"].to(device)
        model.input_limits = input_limits
        model.additional_input_limits = additional_input_limits
        model.output_limits = output_limits
        model = model.to(device)
        return model


class FullyConvolutionalNeuralNetwork(torch.nn.Module):
    """
    https://arxiv.org/pdf/1611.06455.pdf
    """
    def __init__(self, batch_size, seq_len, input_size, output_size):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=7, padding=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Linear(32, self.output_size)

        self.input_limits = None
        self.additional_input_limits = None
        self.output_limits = None

    def forward(self, x):
        """
        Input shape: (batch_size, seq_len, input_size)
        Output shape: (batch_size, 32)
        """
        x = x.permute(0, 2, 1)
        x = self.conv_net(x)
        x = torch.nn.AvgPool1d(kernel_size=self.seq_len)(x)    # Reduce time dimension to 1
        x = x.squeeze() # Remove redundant time dimension
        return self.output_layer(x)

    def save(self, archive_dir, filename=None):
        print("Saving model to {}".format(archive_dir))
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        filename = filename if filename is not None else "FullyConvolutionalNeuralNetwork.pt"
        torch.save({
            "state_dict": self.state_dict(),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "input_size" : self.input_size,
            "output_size" : self.output_size,
            "input_limits": self.input_limits,
            "additional_input_limits": self.additional_input_limits,
            "output_limits": self.output_limits
        }, os.path.join(archive_dir, filename))

    @staticmethod
    def load(archive_file, device):
        print("Loading model from {}".format(archive_file))
        checkpoint = torch.load(archive_file, map_location=device)
        model = FullyConvolutionalNeuralNetwork(checkpoint["batch_size"], checkpoint["seq_len"],
                                                checkpoint["input_size"], checkpoint["output_size"])
        model.load_state_dict(checkpoint["state_dict"])
        input_limits = checkpoint["input_limits"].to(device)
        additional_input_limits = checkpoint["additional_input_limits"].to(device)
        output_limits = checkpoint["output_limits"].to(device)
        model.input_limits = input_limits
        model.additional_input_limits = additional_input_limits
        model.output_limits = output_limits
        model = model.to(device)
        return model


class ResidualNetwork(torch.nn.Module):
    """
    https://arxiv.org/pdf/1611.06455.pdf
    """
    class ResidualBlock(torch.nn.Module):
        def __init__(self, input_size, out_channels):
            super().__init__()
            self.conv_net = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=input_size, out_channels=out_channels, kernel_size=7, padding=3),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.ReLU()
            )

        def forward(self, x):
            y = self.conv_net(x)
            z = y + x
            return torch.nn.ReLU()(z)

    def __init__(self, batch_size, seq_len, input_size, output_size):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.inner_size = 32
        self.fcn_1 = torch.nn.Linear(input_size, self.inner_size)
        self.residual_blocks = torch.nn.Sequential(
            self.ResidualBlock(self.inner_size, self.inner_size),
            self.ResidualBlock(self.inner_size, self.inner_size),
            self.ResidualBlock(self.inner_size, self.inner_size)
        )
        self.output_layer = torch.nn.Linear(self.inner_size, self.output_size)
        self.input_limits = None
        self.additional_input_limits = None
        self.output_limits = None

    def forward(self, x):
        """
        Input shape: (batch_size, seq_len, input_size)
        Output shape: (batch_size, 32)
        """
        x = torch.nn.ReLU()(self.fcn_1(x))
        x = x.permute(0, 2, 1)
        x = self.residual_blocks(x)
        x = torch.nn.AvgPool1d(kernel_size=self.seq_len)(x)    # Reduce time dimension to 1
        x = x.squeeze()      # Remove redundant time dimension
        return self.output_layer(x)

    def save(self, archive_dir, filename=None):
        print("Saving model to {}".format(archive_dir))
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        filename = filename if filename is not None else "ResidualNetwork.pt"
        torch.save({
            "state_dict": self.state_dict(),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "input_size" : self.input_size,
            "output_size" : self.output_size,
            "input_limits": self.input_limits,
            "additional_input_limits": self.additional_input_limits,
            "output_limits": self.output_limits
        }, os.path.join(archive_dir, filename))

    @staticmethod
    def load(archive_file, device):
        print("Loading model from {}".format(archive_file))
        checkpoint = torch.load(archive_file, map_location=device)
        model = ResidualNetwork(checkpoint["batch_size"], checkpoint["seq_len"], checkpoint["input_size"],
                                checkpoint["output_size"])
        model.load_state_dict(checkpoint["state_dict"])
        input_limits = checkpoint["input_limits"].to(device)
        additional_input_limits = checkpoint["additional_input_limits"].to(device)
        output_limits = checkpoint["output_limits"].to(device)
        model.input_limits = input_limits
        model.additional_input_limits = additional_input_limits
        model.output_limits = output_limits
        model = model.to(device)
        return model


class Encoder(torch.nn.Module):
    """
    https://arxiv.org/pdf/1805.03908.pdf
    """
    def __init__(self, batch_size, seq_len, input_size, output_size):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=5, padding=2),
            torch.nn.InstanceNorm1d(128),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=11, padding=5),
            torch.nn.InstanceNorm1d(256),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=21, padding=10),
            torch.nn.InstanceNorm1d(512),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.2)
        )
        self.dense = torch.nn.Linear(256, 32)
        self.inst_norm = torch.nn.InstanceNorm1d(32)
        self.output_layer = torch.nn.Linear(32, self.output_size)

        self.input_limits = None
        self.additional_input_limits = None
        self.output_limits = None

    def forward(self, x):
        """
        Input shape: (batch_size, seq_len, input_size)
        Output shape: (batch_size, output_size)
        """
        x = x.permute(0, 2, 1)
        x = self.conv_net(x)
        # Attention
        attention_weights = torch.nn.Softmax(dim=-1)(x[:, :256, :])   # Softmaxed fist half of filters over trajectory
        weighted_sequence = x[:, 256:, :] * attention_weights
        attention_out = weighted_sequence.sum(dim=-1)
        y = torch.nn.ReLU()(self.dense(attention_out))
        return self.output_layer(y)

    def save(self, archive_dir, filename=None):
        print("Saving model to {}".format(archive_dir))
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        filename = filename if filename is not None else "Encoder.pt"
        torch.save({
            "state_dict": self.state_dict(),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "input_size" : self.input_size,
            "output_size" : self.output_size,
            "input_limits": self.input_limits,
            "additional_input_limits": self.additional_input_limits,
            "output_limits": self.output_limits
        }, os.path.join(archive_dir, filename))

    @staticmethod
    def load(archive_file, device):
        print("Loading model from {}".format(archive_file))
        checkpoint = torch.load(archive_file, map_location=device)
        model = Encoder(checkpoint["batch_size"], checkpoint["seq_len"], checkpoint["input_size"], checkpoint["output_size"])
        model.load_state_dict(checkpoint["state_dict"])
        input_limits = checkpoint["input_limits"].to(device)
        additional_input_limits = checkpoint["additional_input_limits"].to(device)
        output_limits = checkpoint["output_limits"].to(device)
        model.input_limits = input_limits
        model.additional_input_limits = additional_input_limits
        model.output_limits = output_limits
        model = model.to(device)
        return model


class TimeLeNet(torch.nn.Module):
    def __init__(self, batch_size, seq_len, input_size, output_size):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.conv_1 = torch.nn.Conv1d(in_channels=input_size, out_channels=5, kernel_size=5, padding=2)
        self.mp_1 = torch.nn.MaxPool1d(kernel_size=2)       # Divides seq_len by 2
        self.conv_2 = torch.nn.Conv1d(in_channels=5, out_channels=20, kernel_size=5, padding=2)
        self.mp_2 = torch.nn.MaxPool1d(kernel_size=4)        # Divides seq_len by 4
        self.fcn_1 = torch.nn.Linear(int(seq_len / 8), 1)
        self.fcn_2 = torch.nn.Linear(20, 32)
        self.output_layer = torch.nn.Linear(32, self.output_size)

        self.input_limits = None
        self.additional_input_limits = None
        self.output_limits = None

    def forward(self, x):
        """
        Input shape: (batch_size, seq_len, input_size)
        Output shape: (batch_size, 32)
        """
        x = x.permute(0, 2, 1)
        x = torch.nn.ReLU()(self.conv_1(x))
        x = self.mp_1(x)
        x = torch.nn.ReLU()(self.conv_2(x))
        x = self.mp_2(x)
        x = torch.nn.ReLU()(self.fcn_1(x))       # Reduces time dimension to 1
        y = x.squeeze()         # Removes time dimension
        y = torch.nn.ReLU()(self.fcn_2(y))
        return self.output_layer(y)

    def save(self, archive_dir, filename=None):
        print("Saving model to {}".format(archive_dir))
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        filename = filename if filename is not None else "TimeLeNet.pt"
        torch.save({
            "state_dict": self.state_dict(),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "input_size" : self.input_size,
            "output_size" : self.output_size,
            "input_limits": self.input_limits,
            "additional_input_limits": self.additional_input_limits,
            "output_limits": self.output_limits
        }, os.path.join(archive_dir, filename))

    @staticmethod
    def load(archive_file, device):
        print("Loading model from {}".format(archive_file))
        checkpoint = torch.load(archive_file, map_location=device)
        model = TimeLeNet(checkpoint["batch_size"], checkpoint["seq_len"], checkpoint["input_size"], checkpoint["output_size"])
        model.load_state_dict(checkpoint["state_dict"])
        input_limits = checkpoint["input_limits"].to(device)
        additional_input_limits = checkpoint["additional_input_limits"].to(device)
        output_limits = checkpoint["output_limits"].to(device)
        model.input_limits = input_limits
        model.additional_input_limits = additional_input_limits
        model.output_limits = output_limits
        model = model.to(device)
        return model
