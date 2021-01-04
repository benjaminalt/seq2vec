import argparse
import os

import torch
from neural_templates.data.dataset import DirectoryDataset
from torch.utils.data import DataLoader
from utils import transformations

from seq2vec.baselines import model as models
from seq2vec.utils import compute_f1, num_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, augmented_input_sequence):
    with torch.no_grad():
        return model(augmented_input_sequence)


def main(args):
    test_data = DirectoryDataset(args.data_dir)
    data_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=True)

    model_type_name = os.path.basename(args.model_file).split(".")[0]
    model = getattr(models, model_type_name).load(args.model_file, device).to(device)
    model.eval()

    print(f"{num_parameters(model)} parameters")

    all_outputs = []
    all_labels = []
    for environment_inputs, inputs, real, sim, start_states in data_loader:
        sim = sim.to(device)
        inputs_scaled = transformations.scale(sim, *model.input_limits, -1, 1)
        additional_inputs = torch.cat((inputs, environment_inputs), dim=-1).to(device)
        additional_inputs_scaled = transformations.scale(additional_inputs, *model.additional_input_limits, -1, 1)
        augmented_input_seq = torch.cat(
            (inputs_scaled, additional_inputs_scaled.unsqueeze(1).repeat(1, inputs_scaled.size(1), 1)), dim=-1)
        labels = real[:, -1, 1].unsqueeze(-1)
        output_batch = evaluate(model, augmented_input_seq)
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
