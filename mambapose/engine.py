"""_summary_."""

import datetime
import pathlib
import pickle
import shlex
import subprocess
from math import inf

import torch
import utils
from data_setup import ThreeDPW
from PIL import ImageDraw
from tqdm.auto import tqdm

import numpy as np


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    train_loss = 0
    progress_bar = tqdm(total=(len(dataloader)))

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        progress_bar.update(1)

        progress_bar.desc = f"Train loss: {loss:.4f}"

    train_loss = train_loss / len(dataloader)

    return train_loss


def validation_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    validation_loss = 0

    with torch.inference_mode():
        progress_bar = tqdm(total=(len(dataloader)))

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            validation_pred_logits = model(X)

            loss = loss_fn(validation_pred_logits, y)
            validation_loss += loss.item()

            progress_bar.update(1)

            progress_bar.desc = f"Validation loss: {loss:.4f}"

    validation_loss = validation_loss / len(dataloader)
    return validation_loss


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    test_loss = 0

    curr = 0

    test_data = ThreeDPW("test", protocol="train-test", transform=None)

    with torch.inference_mode():
        progress_bar = tqdm(total=(len(dataloader)))

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            samples = []

            for i in range(len(y)):
                samples.append(test_data[curr + i])

            test_pred_logits = model(X)
            test_pred_logits_scaled = test_pred_logits.to("cpu").numpy()

            for sample_idx, sample in enumerate(samples):
                for i in range(len(test_pred_logits_scaled[sample_idx])):
                    if (i + 1) % 2 == 1:
                        test_pred_logits_scaled[sample_idx][i] = (
                            test_pred_logits_scaled[sample_idx][i] * sample["width"]
                        )
                    elif (i + 1) % 2 == 0:
                        test_pred_logits_scaled[sample_idx][i] = (
                            test_pred_logits_scaled[sample_idx][i] * sample["height"]
                        )

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            curr += len(y)

            for sample_idx, sample in enumerate(samples):
                img = sample["images"][2]
                draw = ImageDraw.Draw(sample["images"][2])
                for i in range(18):
                    point_x = test_pred_logits_scaled[sample_idx][2 * i]
                    point_y = test_pred_logits_scaled[sample_idx][2 * i + 1]
                    for j in range(25):
                        for k in range(25):
                            draw.point(
                                (
                                    point_x - 1 + j,
                                    test_pred_logits_scaled[sample_idx][2 * i + 1]
                                    - 1
                                    + k,
                                ),
                                "red",
                            )
                img.save(
                    f"results/images/{datetime.datetime.now()}_{sample['sequence_name']}_{sample['img_idx']}.png",
                    "PNG",
                )
                np.savez(
                    f"results/sequences/{datetime.datetime.now()}_{sample['sequence_name']}_{sample['img_idx']}.npz",
                    y=y[sample_idx].to("cpu").numpy(),
                    y_scaled=np.array(sample["joint_position"]),
                    pred=test_pred_logits[sample_idx].to("cpu").numpy(),
                    pred_scaled=test_pred_logits_scaled[sample_idx],
                )

            progress_bar.update(1)

            progress_bar.desc = f"Test loss: {loss:.4f}"

    test_loss = test_loss / len(dataloader)
    return test_loss


def train(
    model: torch.nn.Module,
    model_name: str,
    train_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    patience: int = 3,
) -> dict[str, list]:
    results = {
        "train_loss": [],
        "validation_loss": [],
    }

    patience_counter = 0
    min_validation_loss = inf
    best_model_epoch = 0

    progress_bar = tqdm(total=epochs)

    for epoch in range(epochs):
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        validation_loss = validation_step(
            model=model,
            dataloader=validation_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        progress_bar.desc = (
            f"Train_loss: {train_loss:.4f} | validation_loss: {validation_loss:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["validation_loss"].append(validation_loss)

        if validation_loss <= min_validation_loss:
            model_save_path = pathlib.Path(__file__).parent.parent.absolute() / "models"

            if epoch != 0:
                subprocess.run(
                    shlex.split(
                        f"rm {model_save_path / model_name}-{best_model_epoch + 1}.pth"
                    )
                )

            utils.save_model(
                model=model,
                target_path=model_save_path,
                model_name=f"{model_name}-{epoch + 1}.pth",
            )

            patience_counter = 0
            min_validation_loss = validation_loss
            best_model_epoch = epoch

        else:
            patience_counter += 1

        with open(
            pathlib.Path(__file__).parent.parent.absolute()
            / "metrics"
            / f"vim-{epoch + 1}.pkl",
            "wb",
        ) as file:
            pickle.dump(results, file)

        if epoch != 0:
            to_delete = (
                pathlib.Path(__file__).parent.parent.absolute()
                / "metrics"
                / f"vim-{epoch}.pkl"
            )

            subprocess.run(shlex.split(f"rm {to_delete}"))

        progress_bar.update(1)

        if patience_counter >= patience:
            break

    model_save_path = pathlib.Path(__file__).parent.parent.absolute() / "models"

    model.load_state_dict(
        torch.load(f"{model_save_path / model_name}-{best_model_epoch + 1}.pth")
    )

    test_loss = test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
    )

    results["test_loss"] = test_loss

    return results
