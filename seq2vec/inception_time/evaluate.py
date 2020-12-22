import argparse

import torch
from neural_templates.data.dataset import DirectoryDataset
from torch.utils.data import TensorDataset, DataLoader
from utils import transformations

from seq2vec.data import load_dataset
from seq2vec.inception_time.model import InceptionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_f1(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred = torch.round(y_pred)
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return precision, recall, f1


def evaluate(model, sequence, params):
    with torch.no_grad():
        return model(sequence, params)


def main(args):
    test_data = DirectoryDataset(args.data_dir)
    data_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=True)

    model = InceptionModel.load(args.model_file, device).to(device)
    model.eval()

    all_outputs = []
    all_labels = []
    for environment_inputs, inputs, real, sim, start_states in data_loader:
        inputs_scaled = transformations.scale(sim.to(device), *model.input_limits, -1, 1)
        additional_inputs = torch.cat((inputs, environment_inputs), dim=-1)
        additional_inputs_scaled = transformations.scale(additional_inputs.to(device), *model.additional_input_limits, -1, 1).to(
            device)
        labels = real[:, -1, 1].unsqueeze(-1)
        output_batch = evaluate(model, inputs_scaled, additional_inputs_scaled)
        # outputs are in range (-1, 1) --> scale to (0, 1)
        outputs_unscaled = transformations.scale(output_batch, -1, 1, *model.output_limits).cpu()
        outputs_unscaled = torch.clamp(outputs_unscaled, 0, 1)
        all_outputs.append(outputs_unscaled)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    precision, recall, f1 = compute_f1(all_outputs, all_labels)
    print(f"Precision: {precision.item():.4f}, recall: {recall.item():.4f}, F1: {f1.item():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=str)
    parser.add_argument("data_dir", type=str)
    main(parser.parse_args())
