from data_setup import (
    ConcatImages,
    Depth,
    DictToXY,
    NoneToTensor,
    NormalizeRGB,
    OpticalFlow,
    PILToTensor,
    Scale,
    ThreeDPW,
)
from torchvision.transforms import v2
import torch
from PIL import Image, ImageDraw
from pathlib import Path

from vision_mamba import Vim
from tqdm import tqdm
import pickle

import os

from torch.utils.data import DataLoader

DATA_PATH = Path("data")
SPLIT = "validation"
SEQUENCES_PATH = DATA_PATH / "sequences" / SPLIT
IMAGES_PATH = DATA_PATH / "images"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Vim(
    dim=256,
    dt_rank=256,
    dim_inner=256,
    d_state=256,
    num_classes=36,
    image_size=224,
    patch_size=32,
    channels=25,
    dropout=0.5,
    depth=20,
).to(device)

model.load_state_dict(torch.load("models/vim-11.pth"))

model.eval()

transform = v2.Compose(
    [
        Scale(224),
        PILToTensor(),
        OpticalFlow(),
        NormalizeRGB(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        NoneToTensor(),
        ConcatImages(),
        DictToXY(),
    ]
)

test_data = ThreeDPW(SPLIT, protocol="train-test", transform=transform)
test_dataloader = DataLoader(
    test_data,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

results: dict[str, list[torch.tensor]] = {}
total_loss = 0
loss_fn = torch.nn.MSELoss()

with torch.inference_mode(mode=True):
    for batch, (X, y) in tqdm(enumerate(test_dataloader)):

        y = y.to(device)

        # sequence_name = sample["sequence_name"]
        # if sequence_name == "downtown_downstairs_00":
        #     continue
        # width = sample["width"]
        # height = sample["height"]
        # img_idx = sample["img_idx"]

        # with open(SEQUENCES_PATH / f"{sequence_name}.pkl", "rb") as f_obj:
        #     data = pickle.load(f_obj, encoding="latin1")

        y_pred = model(X) * 4500

        # img = Image.open(IMAGES_PATH / sequence_name / f"image_{img_idx:05d}.jpg")
        # initial_width, initial_height = img.size

        # print(data["poses2d"][0][0])
        # draw = ImageDraw.Draw(img)
        # for i in range(17):
        #     draw.text((data["poses2d"][0][0][0][i], data["poses2d"][0][0][1][i]), str(data["poses2d"][0][0][2][i]))
        # img.save("test.png", "PNG")
        
        # print(y_pred)
        # for i in range(len(y_pred)):
        #     if (i + 1) % 2 == 1:
        #         y_pred[i] = y_pred[i] / width * initial_width
        #         y[i] = y[i] / width * initial_width
        #     elif (i + 1) % 2 == 0:
        #         y_pred[i] = y_pred[i] / height * initial_height
        #         y[i] = y[i] / height * initial_height
        print(f"{loss_fn(y_pred, y):.4f}")
        # print(31 * "=")


