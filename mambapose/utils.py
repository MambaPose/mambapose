"""_summary_."""

from pathlib import Path

import requests
import torch
from tqdm import tqdm


def save_model(model: torch.nn.Module, target_path: Path, model_name: str):

    target_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_path = target_path / model_name

    torch.save(obj=model.state_dict(), f=model_path)

def download_file(url, path):

    TIMEOUT = 60 # seconds

    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True, timeout=TIMEOUT)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.desc = path.name
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError(f"Could not download file from {url} to {path}.")
