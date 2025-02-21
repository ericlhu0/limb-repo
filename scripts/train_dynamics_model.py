"""Script to train a dynamics model."""

import argparse
import os
import shutil
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb
from limb_repo.dynamics.models.learned_dynamics import (
    NeuralNetworkConfig,
    PyTorchLearnedDynamicsModel,
)
from limb_repo.model_training.training.learned_dynamics_dataset import (
    LearnedDynamicsDataset,
)
from limb_repo.utils import file_utils, utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()
    run_name = args.run_name + "_" + time.strftime("%Y-%m-%d_%H-%M-%S")

    # files!
    nn_config_path = "assets/configs/nn_configs/30-512-512-512-512-12.yaml"
    data_path = "/home/eric/lr-dir/limb-repo/_the_good_stuff/combined.hdf5"
    weights_dir = f"_weights/{run_name}/"
    nn_config_path = file_utils.to_abs_path(nn_config_path)
    data_path = file_utils.to_abs_path(data_path)
    weights_dir = file_utils.to_abs_path(weights_dir)

    # # combine datasets
    # hdf5_saver = file_utils.HDF5Saver(
    #     file_utils.to_abs_path(f"_the_good_stuff/"),
    #     file_utils.to_abs_path("_out/temp/"),
    # )
    # print("combining files")
    # data_path = hdf5_saver.combine_temp_hdf5s(
    # data_dirs=["01-16_02-07-00", "01-16_02-07-02", "01-16_02-07-04", "01-16_02-07-07"]
    # )

    os.makedirs(weights_dir, exist_ok=True)
    shutil.copy2(nn_config_path, weights_dir)

    wandb.login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nn_config = utils.parse_config(nn_config_path, NeuralNetworkConfig)
    model = PyTorchLearnedDynamicsModel(nn_config)

    # Use DataParallel to wrap the model
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    # model.load_state_dict(torch.load('/home/eric/lr-dir/limb-repo/_weights/... .pth'))
    model.to(device)

    batch_size = 2**12
    learning_rate = 1e-3
    epochs = 500
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-4
    )  # Initialize the scheduler

    dataset = LearnedDynamicsDataset(data_path)

    print("Normalized inputs with:")
    print("  max:", dataset.max_features)
    print("  min:", dataset.min_features)
    print("  range:", dataset.range_features)
    print("  std: ", dataset.std_features)

    print("Normalized outputs with:")
    print("  max:", dataset.max_labels)
    print("  min:", dataset.min_labels)
    print("  range:", dataset.range_labels)
    print("  std: ", dataset.std_labels)

    total_size = len(dataset)
    test_ratio = 0.2
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size

    # Split the dataset deterministically
    print("start splitting dataset")
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))
    print("done splitting dataset")

    # # Create DataLoaders
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=10,
    # )
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=10,
    # )

    # Do not use num_workers when data is already on GPU
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size, shuffle=False, drop_last=True
    )

    print("train data len:", len(train_dataset))
    print("epochs:", epochs)

    run_config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "model_arch": str(model),
        "train_dataset_length": len(train_dataset),
        "batch_size": batch_size,
        "human_n_dofs": 6,
        "min_features": dataset.min_features,
        "max_features": dataset.max_features,
        "range_features": dataset.range_features,
        "min_labels": dataset.min_labels,
        "max_labels": dataset.max_labels,
        "range_labels": dataset.range_labels,
        "labels_normalization": "tanh(0.125 * x)",
    }

    run = wandb.init(
        # Set the project where this run will be logged
        project="new_dynamics",
        name=run_name,
        config=run_config,
    )

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        # train loop
        train_size = len(train_dataloader)
        model.train()
        all_train_predictions: list[torch.Tensor] = []
        all_train_labels: list[torch.Tensor] = []

        for batch, (features, labels) in enumerate(train_dataloader):
            # Move data to GPU
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            preds = model(features)
            all_train_predictions.append(preds)
            all_train_labels.append(labels)

            # Backprop
            loss = loss_fn(preds, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        train_mse_loss = loss_fn(
            torch.cat(all_train_predictions), torch.cat(all_train_labels)
        )

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), weights_dir + f"/model_weights_{epoch}.pth")

        # testing loop
        test_mse_loss = torch.tensor(0.0).to(device)
        all_diffs = torch.tensor([]).to(device)
        all_test_predictions = []
        all_test_labels = []

        model.eval()

        with torch.no_grad():
            for batch, (features, labels) in enumerate(test_dataloader):
                # Move data to GPU
                features, labels = features.to(device), labels.to(device)

                pred = model(features)
                all_test_predictions.append(pred)
                all_test_labels.append(labels)

                batch_diff = pred - labels
                all_diffs = torch.cat((all_diffs, batch_diff))

        test_mse_loss = loss_fn(torch.cat(all_test_predictions), all_test_labels)

        # all_diffs = np.array(all_diffs)
        avg_diff = torch.mean(torch.abs((all_diffs)))
        max_diff = torch.max(torch.abs((all_diffs)))
        max_location = torch.floor(
            torch.argmax(torch.abs((all_diffs))) / all_diffs.shape[1]
        )

        print("Train MSE error:", train_mse_loss)
        print("Test MSE error:", test_mse_loss)
        print("Mean diff:", avg_diff)
        print("Maxmimum diff:", max_diff)
        print("Max Location", max_location)
        print("Max diff prediction:", all_test_predictions[int(max_location.item())])
        print("Max diff label:", all_test_labels[int(max_location.item())])

        wandb.log(
            {
                "train_mse_loss": train_mse_loss,
                "test_mse_loss": test_mse_loss,
                "avg_diff": avg_diff,
                "max_diff": max_diff,
                "diff_array": wandb.Histogram(
                    list(torch.ravel(all_diffs).cpu().numpy())
                ),
            }
        )

        # Step the scheduler
        scheduler.step()

    torch.save(model.state_dict(), weights_dir + "/model_weights_final.pth")
    print("saved model")
