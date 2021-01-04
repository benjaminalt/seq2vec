import argparse
import math
import os
import time
from datetime import datetime

import torch
from neural_templates.data.dataset import DirectoryDataset
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from utils import transformations

from seq2vec.utils import plot_loss_history, time_since
from seq2vec.baselines import model as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_limits(dataset: Dataset):
    input_limits = None
    additional_input_limits = None
    output_limits = None
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=1)
    for environment_inputs, inputs, real, sim, start_states in tqdm(loader):
        labels = real[:, -1, 1].unsqueeze(-1) # Success probability
        additional_inputs = torch.cat((inputs, environment_inputs), dim=-1)
        sim_flattened = torch.flatten(sim, end_dim=-2)
        batch_input_limits = torch.stack((torch.min(sim_flattened, dim=0)[0], torch.max(sim_flattened, dim=0)[0]))
        batch_additional_input_limits = torch.stack((torch.min(additional_inputs, dim=0)[0], torch.max(additional_inputs, dim=0)[0]))
        batch_output_limits = torch.stack((torch.min(labels, dim=0)[0], torch.max(labels, dim=0)[0]))
        if input_limits is None:
            input_limits = batch_input_limits
            additional_input_limits = batch_additional_input_limits
            output_limits = batch_output_limits
        else:
            input_limits = torch.stack((torch.min(batch_input_limits[0], input_limits[0]),
                                             torch.max(batch_input_limits[1], input_limits[1])))
            additional_input_limits = torch.stack((torch.min(batch_additional_input_limits[0], additional_input_limits[0]),
                                           torch.max(batch_additional_input_limits[1], additional_input_limits[1])))
            output_limits = torch.stack((torch.min(batch_output_limits[0], output_limits[0]),
                                              torch.max(batch_output_limits[1], output_limits[1])))

    # If min and max are identical, this causes trouble when scaling --> division by zero produces NaN
    # Set min to val - 1 and max to val + 1
    def disambiguate_identical_limits(limits: torch.Tensor):
        for dim_idx in range(limits.size(1)):
            if limits[0, dim_idx] == limits[1, dim_idx]:
                orig_value = limits[0, dim_idx].clone()
                limits[0, dim_idx] = orig_value - 1  # new min
                limits[1, dim_idx] = orig_value + 1  # new max

    disambiguate_identical_limits(input_limits)
    disambiguate_identical_limits(additional_input_limits)
    disambiguate_identical_limits(output_limits)
    return input_limits, additional_input_limits, output_limits


def train(dataset_path: str, baseline_type: str, batch_size, num_epochs, learning_rate):
    all_data = DirectoryDataset(dataset_path)
    input_limits, additional_input_limits, output_limits = compute_limits(all_data)
    split_idx = int(0.9 * len(all_data))
    training_data = DirectoryDataset(dataset_path, end=split_idx)
    validation_data = DirectoryDataset(dataset_path, start=split_idx)
    print("Training/validation split: {}/{}".format(len(training_data), len(validation_data)))
    train_loader = DataLoader(training_data, batch_size=batch_size, drop_last=False, pin_memory=True, num_workers=1)
    validate_loader = DataLoader(validation_data, batch_size=batch_size, drop_last=False, pin_memory=True,
                                 num_workers=1)

    input_dim = input_limits.size(-1) + additional_input_limits.size(-1)
    output_dim = output_limits.size(-1)
    model_class = getattr(models, baseline_type)
    seq_len = 250
    model = model_class(batch_size, seq_len, input_dim, output_dim)

    model.input_limits = input_limits
    model.additional_input_limits = additional_input_limits
    model.output_limits = output_limits

    output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, "output", "baselines")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loss_history = train_loop(model, train_loader, validate_loader, num_epochs, learning_rate)
    timestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    model.save(output_dir)
    plot_loss_history(loss_history, os.path.join(output_dir, "{}_{}_loss.png".format(baseline_type, timestamp)),
                      labels=["Train", "Validate"])


def train_loop(model, train_loader, val_loader, n_epochs, learning_rate):
    start = time.time()
    loss_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    model = model.to(device)

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        model.train()
        for environment_inputs, inputs, real, sim, start_states in tqdm(train_loader):
            inputs_scaled = transformations.scale(sim, *model.input_limits, -1, 1).to(device)
            additional_inputs = torch.cat((inputs, environment_inputs), dim=-1)
            additional_inputs_scaled = transformations.scale(additional_inputs, *model.additional_input_limits, -1, 1).to(device)
            augmented_input_seq = torch.cat((inputs_scaled, additional_inputs_scaled.unsqueeze(1).repeat(1, inputs_scaled.size(1), 1)), dim=-1)
            labels = real[:, -1, 1].unsqueeze(-1).to(device)
            # labels_scaled = transformations.scale(labels, *model.output_limits, -1, 1).to(device)
            batch_loss = train_step(augmented_input_seq, labels, model, optimizer, criterion, validate=False)
            train_loss += batch_loss

        avg_train_loss = train_loss / len(train_loader)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for environment_inputs, inputs, real, sim, start_states in tqdm(val_loader):
                inputs_scaled = transformations.scale(sim, *model.input_limits, -1, 1).to(device)
                additional_inputs = torch.cat((inputs, environment_inputs), dim=-1)
                additional_inputs_scaled = transformations.scale(additional_inputs, *model.additional_input_limits, -1,
                                                                 1).to(device)
                augmented_input_seq = torch.cat(
                    (inputs_scaled, additional_inputs_scaled.unsqueeze(1).repeat(1, inputs_scaled.size(1), 1)),
                    dim=-1)
                labels = real[:, -1, 1].unsqueeze(-1).to(device)
                # labels_scaled = transformations.scale(labels, *model.output_limits, -1, 1).to(device)
                batch_loss = train_step(augmented_input_seq, labels, model, optimizer, criterion, validate=True)
                val_loss += batch_loss
        avg_val_loss = val_loss / len(val_loader)

        loss_history.append([avg_train_loss, avg_val_loss])

        print('%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                     epoch, epoch / n_epochs * 100, avg_val_loss))
    return loss_history


def train_step(inputs, labels, model, optimizer, criterion, validate=False):
    output = model(inputs)
    loss = criterion(output, labels)
    if not validate:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def main(args):
    batch_size = 64
    learning_rate = 5e-5
    num_epochs = 50
    train(args.dataset_path, args.baseline_type, batch_size, num_epochs, learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("baseline_type", type=str)
    main(parser.parse_args())
