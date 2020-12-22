import argparse
import math
import os
import time
from datetime import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from seq2vec.data import load_dataset
from seq2vec.inception_time.model import InceptionModel
from seq2vec.utils import plot_loss_history, time_since

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset_path, batch_size, num_epochs, learning_rate):
    """
    Train an InceptionTime network on "inverse frequency modulation": Given a modulated signal and a set of inputs,
    find the original frequency of the carrier signal
    """
    normalized_carrier_params_train, normalized_carrier_train, normalized_signal_params_train, normalized_modulated_train, _, _ = load_dataset(dataset_path,
                                                                                                       kind="train")
    train_dataset = TensorDataset(torch.from_numpy(normalized_carrier_params_train).float(),
                                  torch.from_numpy(normalized_carrier_train).float(),
                                  torch.from_numpy(normalized_signal_params_train).float(),
                                  torch.from_numpy(normalized_modulated_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                              pin_memory=True, drop_last=True)

    normalized_carrier_params_validate, normalized_carrier_validate, normalized_signal_params_validate, normalized_modulated_validate, _, _ = load_dataset(
        dataset_path, kind="validate")
    validate_dataset = TensorDataset(torch.from_numpy(normalized_carrier_params_validate).float(),
        torch.from_numpy(normalized_carrier_validate).float(),
                                     torch.from_numpy(normalized_signal_params_validate).float(),
                                     torch.from_numpy(normalized_modulated_validate).float())
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                 pin_memory=True, drop_last=True)
    input_dim = normalized_modulated_train.shape[-1]
    additional_input_dim = normalized_signal_params_train.shape[-1]
    output_dim = normalized_carrier_params_train.shape[-1]
    print(f"Input dim: {input_dim}, additional input dim: {additional_input_dim}, output dim: {output_dim}")
    model = InceptionModel(num_blocks=9, in_channels=input_dim, out_channels=32, bottleneck_channels=16, kernel_sizes=32,
                           use_residuals=True, additional_input_dim=additional_input_dim, output_dim=output_dim)
    output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, "output", "inception_time")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    loss_history = train_loop(model, train_loader, validate_loader, num_epochs, learning_rate)
    timestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    model.save(output_dir, f"{timestamp}_model.pt")
    plot_loss_history(loss_history, os.path.join(output_dir, "{}_loss.png".format(timestamp)), labels=["Train", "Validate"])


def train_loop(model, train_loader, val_loader, n_epochs, learning_rate):
    start = time.time()
    loss_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    model = model.to(device)

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        model.train()
        for carrier_params, carrier_sig, signal_params, modulated_sig in tqdm(train_loader):
            carrier_params = carrier_params.to(device)
            carrier_sig = carrier_sig.to(device)
            signal_params = signal_params.to(device)
            modulated_sig = modulated_sig.to(device)
            batch_loss = train_step(carrier_params, carrier_sig, signal_params, modulated_sig, model, optimizer, criterion, validate=False)
            train_loss += batch_loss

        avg_train_loss = train_loss / len(train_loader)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for carrier_params, carrier_sig, signal_params, modulated_sig in tqdm(val_loader):
                carrier_params = carrier_params.to(device)
                carrier_sig = carrier_sig.to(device)
                signal_params = signal_params.to(device)
                modulated_sig = modulated_sig.to(device)
                batch_loss = train_step(carrier_params, carrier_sig, signal_params, modulated_sig, model, optimizer, criterion, validate=True)
                val_loss += batch_loss
        avg_val_loss = val_loss / len(val_loader)

        loss_history.append([avg_train_loss, avg_val_loss])

        print('%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                     epoch, epoch / n_epochs * 100, avg_val_loss))
    return loss_history


def train_step(carrier_params, carrier_sig, signal_params, modulated_signal, model, optimizer, criterion, validate=False):
    output = model(modulated_signal, signal_params)
    loss = criterion(output, carrier_params)
    if not validate:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def main(args):
    batch_size = 64
    learning_rate = 5e-5
    num_epochs = 10
    train(args.dataset_path, batch_size, num_epochs, learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    main(parser.parse_args())
