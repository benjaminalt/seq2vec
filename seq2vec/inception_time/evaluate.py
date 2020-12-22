import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader

from seq2vec.data import load_dataset
from seq2vec.inception_time.model import InceptionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, sequence, params):
    with torch.no_grad():
        return model(sequence, params)


def main(args):
    normalized_carrier_params, normalized_carrier, normalized_signal_params, normalized_modulated, _, _ = load_dataset(
        args.data_dir, "test")
    model = InceptionModel.load(args.model_file, device).to(device)
    model.eval()

    dataset = TensorDataset(torch.from_numpy(normalized_carrier_params).float(),
                            torch.from_numpy(normalized_carrier).float(),
                            torch.from_numpy(normalized_signal_params).float(),
                            torch.from_numpy(normalized_modulated).float())
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=True)
    loss_fn = torch.nn.MSELoss()
    loss = 0
    for carrier_params, carrier_sig, signal_params, modulated_sig in data_loader:
        carrier_sig = carrier_sig.to(device)
        signal_params = signal_params.to(device)
        modulated_sig = modulated_sig.to(device)
        output_batch = evaluate(model, modulated_sig, signal_params)
        for i in range(len(output_batch)):
            print("---" * 10)
            print(carrier_params[i].cpu().tolist())
            print(output_batch[i].cpu().tolist())
        loss += loss_fn(output_batch, carrier_params).item()
    loss /= len(data_loader)
    print("Test loss: {}".format(loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=str)
    parser.add_argument("data_dir", type=str)
    main(parser.parse_args())
