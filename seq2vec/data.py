import argparse
import os

import joblib
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from seq2seq_transduction.utils import plot_waves

script_dir = os.path.abspath(os.path.dirname(__file__))


def generate_sins(n, seq_len, amplitude_limits=None, freq_limits=None, phase_limits=None):
    if phase_limits is None:
        phase_limits = [0, 2 * np.pi]
    if freq_limits is None:
        freq_limits = [1, 5]
    if amplitude_limits is None:
        amplitude_limits = [0.1, 5]
    params = []
    data = []
    for _ in range(n):
        amp = random.uniform(*amplitude_limits)
        freq = random.uniform(*freq_limits)
        phase = random.uniform(*phase_limits)
        x = np.arange(seq_len)
        y = amp * np.sin(2 * np.pi * freq * (x / seq_len) + phase)
        params.append([amp, freq, phase])
        data.append(y)
    return np.array(params), np.expand_dims(np.array(data), -1)


def frequency_modulate(carriers, carrier_params, signal_params):
    seq_len = carriers.shape[1]
    carrier_freqs = np.expand_dims(carrier_params[:,1], -1)
    signal_freqs = np.expand_dims(signal_params[:,1], -1)
    t = np.linspace(0, 1, seq_len)
    modulated = np.sin(2 * np.pi * carrier_freqs * t + 1.0 * np.sin(2 * np.pi * signal_freqs * t))
    return np.expand_dims(modulated, -1)


def normalize(data, scaler=None):
    orig_shape = data.shape
    if len(data.shape) == 3:
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(data)
    normalized_data = scaler.transform(data).reshape(orig_shape)
    return normalized_data, scaler


def denormalize(data, scaler):
    orig_shape = data.shape
    if len(data.shape) == 3:
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    denormalized_data = scaler.inverse_transform(data).reshape(orig_shape)
    return denormalized_data


def generate_dataset(n, seq_len, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        for kind in ("train", "validate", "test"):
            os.makedirs(os.path.join(output_dir, kind))

    print("Generating dataset in {}...".format(output_dir))
    carrier_params, carriers = generate_sins(n, seq_len)
    signal_params, signals = generate_sins(n, seq_len)
    modulated_signals = frequency_modulate(carriers, carrier_params, signal_params)
    normalized_carrier_params, param_scaler = normalize(carrier_params)
    normalized_carrier, signal_scaler = normalize(carriers)
    normalized_signal_params, _ = normalize(signal_params, param_scaler)
    normalized_modulated, _ = normalize(modulated_signals, signal_scaler)

    np.save(os.path.join(output_dir, "train", "carrier_params.npy"), normalized_carrier_params)
    np.save(os.path.join(output_dir, "train", "carrier.npy"), normalized_carrier)
    np.save(os.path.join(output_dir, "train", "signal_params.npy"), normalized_signal_params)
    np.save(os.path.join(output_dir, "train", "modulated.npy"), normalized_modulated)

    for kind in ("validate", "test"):
        num_data = int(0.2 * n)
        carrier_params, carriers = generate_sins(num_data, seq_len)
        signal_params, signals = generate_sins(num_data, seq_len)
        modulated_signals = frequency_modulate(carriers, carrier_params, signal_params)
        normalized_carrier_params, _ = normalize(carrier_params, param_scaler)
        normalized_carrier, _ = normalize(carriers, signal_scaler)
        normalized_signal_params, _ = normalize(signal_params, param_scaler)
        normalized_modulated, _ = normalize(modulated_signals, signal_scaler)
        np.save(os.path.join(output_dir, kind, "carrier_params.npy"), normalized_carrier_params)
        np.save(os.path.join(output_dir, kind, "carrier.npy"), normalized_carrier)
        np.save(os.path.join(output_dir, kind, "signal_params.npy"), normalized_signal_params)
        np.save(os.path.join(output_dir, kind, "modulated.npy"), normalized_modulated)

    joblib.dump(signal_scaler, os.path.join(output_dir, "signal_scaler.pkl"))
    joblib.dump(param_scaler, os.path.join(output_dir, "param_scaler.pkl"))
    print("Done.")


def load_dataset(dataset_path, kind="train"):
    print("Loading dataset from {}...".format(dataset_path))
    normalized_carrier_params = np.load(os.path.join(dataset_path, kind, "carrier_params.npy"))
    normalized_carrier = np.load(os.path.join(dataset_path, kind, "carrier.npy"))
    normalized_signal_params = np.load(os.path.join(dataset_path, kind, "signal_params.npy"))
    normalized_modulated = np.load(os.path.join(dataset_path, kind, "modulated.npy"))
    signal_scaler = joblib.load(os.path.join(dataset_path, "signal_scaler.pkl"))
    param_scaler = joblib.load(os.path.join(dataset_path, "param_scaler.pkl"))
    print("Done.")
    return normalized_carrier_params, normalized_carrier, normalized_signal_params, normalized_modulated, signal_scaler, param_scaler


def main(args):
    if args.command == "plot":
        for _ in range(args.n):
            carrier_params, carriers = generate_sins(1, args.seq_len, freq_limits=[10, 20])
            signal_params, signals = generate_sins(1, args.seq_len)
            modulated_signals = frequency_modulate(carriers, carrier_params, signal_params)
            fig, axes = plt.subplots(3)
            plot_waves(carrier_params, carriers, ax=axes[0])
            plot_waves(signal_params, signals, ax=axes[1])
            plot_waves(signal_params, modulated_signals, ax=axes[2])
            plt.show()
    elif args.command == "generate_dataset":
        generate_dataset(args.n, args.seq_len, args.output_dir)
    else:
        raise ValueError("Invalid command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plot", "generate_dataset"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "data"))
    main(parser.parse_args())
