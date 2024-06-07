"""_summary_."""

import os
import pathlib
import pickle
import shlex
import subprocess
import zipfile
from collections import namedtuple

import gc
import numpy as np

import torch
from depth_anything.depth_anything import dpt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from torchvision.transforms import functional, v2
from utils import download_file

from tqdm import tqdm

ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
DATA_PATH = ROOT_PATH / "data"
IMAGES_PATH = DATA_PATH / "images"
SEQUENCES_PATH = DATA_PATH / "sequences"

DataPoint = namedtuple("DataPoint", ["sequence_name", "idx", "joint_position"])

device = "cuda" if torch.cuda.is_available() else "cpu"
                
# img_count = 0
# mean = torch.zeros(3)
# std = torch.zeros(3)

class ThreeDPW(Dataset):

    def __init__(
        self,
        split,
        protocol="train-test",
        frames_before=2,
        frames_after=2,
        multi_person=False,
        transform=None,
    ):
        self._download_dataset()
        self.splits = []
        self.normalize_splits = []
        self.data: list = []
        self.frames_before = frames_before
        self.frames_after = frames_after
        self.transform = transform
        multi_person = False  # Forced for now

        match protocol:
            case "all-test":
                match split:
                    case "test":
                        self.splits = ["train", "validation", "test"]
            case "train-test":
                match split:
                    case "train":
                        self.splits = ["train"]
                        self.normalize_splits = ["train"]
                    case "validation":
                        self.splits = ["validation"]
                    case "test":
                        self.splits = ["test"]
            case "validation":
                match split:
                    case "validation":
                        self.splits = ["validation"]
                    case "test":
                        self.splits = ["train", "test"]
            case "all-train":
                match split:
                    case "train":
                        self.splits = ["train", "validation", "test"]
                        self.normalize_splits = ["train", "validation", "test"]

        if not self.splits:
            raise ValueError("Split and protocol doesn't match.")
        
        # global img_count
        # mean_std_calculated = bool(img_count)

        for split in self.splits:
            split_path = SEQUENCES_PATH / split
            for pkl in split_path.glob("*.pkl"):
                if pkl.name == "downtown_downstairs_00.pkl":
                    continue
                with open(pkl, "rb") as f_obj:
                    data = pickle.load(f_obj, encoding="latin1")

                joint_positions = data["poses2d"]

                if not multi_person and len(joint_positions) > 1:
                    continue

                sequence_name = data["sequence"]
                joint_positions = joint_positions[0]

                skip = 0

                for idx, joint_position in enumerate(joint_positions):
                    skip += 1

                    if skip % 2 == 0:
                        continue

                    tmp_joint_position = torch.zeros(36)
                    for lol in range(18):
                        tmp_joint_position[2*lol] = joint_position[0][lol]
                        tmp_joint_position[2*lol + 1] = joint_position[1][lol]
                    joint_position = tmp_joint_position.numpy()

                    self.data.append(
                        DataPoint(
                            sequence_name=sequence_name,
                            idx=idx,
                            joint_position=joint_position,
                        )
                    )


        #         if split in self.normalize_splits and not mean_std_calculated:

        #             global mean
        #             global std

        #             print(f"Getting std and mean for {split}")

        #             for image_path in tqdm((IMAGES_PATH / sequence_name).glob("*.jpg")):
        #                 image = Image.open(image_path)

        #                 image = functional.pil_to_tensor(image).type(torch.float32)
        #                 _, height, width = image.shape
        #                 mean += torch.mean(image, dim=[1, 2])
        #                 std += torch.std(image, dim=[1, 2])

        #                 img_count += 1
                
        # if img_count and not mean_std_calculated:
        #     mean /= img_count
        #     std /= img_count

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sequence_name, img_idx, joint_position = self.data[idx]
        images = []

        for i in range(self.frames_before):
            image = self._load_image(sequence_name, img_idx - (i + 1))
            images.append(image)

        image = self._load_image(sequence_name, img_idx)
        images.append(image)

        channel_count = len(image.getbands())
        width, height = image.size

        for i in range(self.frames_after):
            image = self._load_image(sequence_name, img_idx + (i + 1))
            images.append(image)

        sample = {
            "sequence_name": sequence_name,
            "img_idx": img_idx,
            "images": images,
            "joint_position": joint_position,
            "channel_count": channel_count,
            "width": width,
            "height": height,
        }

        if self.transform:
            sample = self.transform(sample)

        gc.collect()
        torch.cuda.empty_cache()

        return sample

    def _download_dataset(self):
        FILES = (
            (
                "https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip",
                DATA_PATH / "imageFiles.zip",
            ),
            (
                "https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip",
                DATA_PATH / "sequenceFiles.zip",
            ),
        )

        if not DATA_PATH.exists():
            os.mkdir(DATA_PATH)

        if not os.listdir(DATA_PATH):
            self._print_license_agreement()

            for file in FILES:
                download_file(file[0], file[1])

            self._extract_dataset()

    def _extract_dataset(self):

        FILES = (
            DATA_PATH / "imageFiles.zip",
            DATA_PATH / "sequenceFiles.zip",
        )

        with zipfile.ZipFile(FILES[0], "r") as zip_obj:
            zip_obj.extractall(DATA_PATH)

        with zipfile.ZipFile(FILES[1], "r") as zip_obj:
            zip_obj.extractall(DATA_PATH)

        subprocess.run(shlex.split(f"rm {FILES[0]}"))
        subprocess.run(shlex.split(f"rm {FILES[1]}"))
        subprocess.run(shlex.split(f"rm -r {DATA_PATH / '__MACOSX'}"))

        subprocess.run(
            shlex.split(f"mv {DATA_PATH / 'imageFiles'} {DATA_PATH / 'images'}")
        )
        subprocess.run(
            shlex.split(f"mv {DATA_PATH / 'sequenceFiles'} {DATA_PATH / 'sequences'}")
        )

    def _print_license_agreement(self):
        TERM_MIN_COLS = 80
        term_cols = max(TERM_MIN_COLS, os.get_terminal_size().columns)

        TERM_MAX_TEXT = round(term_cols * 3 / 4)
        FILL_CHAR = "#"

        to_print = "By downloading 3DPW dataset, you're agreeing to the license provided at https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html".split(
            " "
        )

        print(term_cols * FILL_CHAR)
        print(FILL_CHAR + (term_cols - 2) * " " + FILL_CHAR)

        to_print_line = ""

        while to_print:
            len_cur = len(to_print[0])
            if len(to_print_line) + len_cur <= TERM_MAX_TEXT:
                if not to_print_line:
                    to_print_line += to_print[0]
                else:
                    to_print_line += f" {to_print[0]}"
                to_print.pop(0)
            else:
                print(FILL_CHAR + to_print_line.center(term_cols - 2) + FILL_CHAR)
                to_print_line = ""

        if to_print_line:
            print(FILL_CHAR + to_print_line.center(term_cols - 2) + FILL_CHAR)

        print(FILL_CHAR + (term_cols - 2) * " " + FILL_CHAR)
        print(term_cols * FILL_CHAR)
        print()

    def _load_image(self, sequence_name, idx):
        try:
            image = Image.open(IMAGES_PATH / sequence_name / f"image_{idx:05d}.jpg")
        except FileNotFoundError:
            image = None
        return image


def get_dataloaders(batch_size: int, num_workers: int = 0):
    if num_workers is None:
        num_workers = os.cpu_count()

    TRAIN_PATH = DATA_PATH / "train"
    VALIDATION_PATH = DATA_PATH / "validation"
    TEST_PATH = DATA_PATH / "test"

    transform = v2.Compose(
        [
            Scale(224),
            PILToTensor(),
            #Depth(),
            OpticalFlow(),
            NormalizeRGB(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            NoneToTensor(),
            ConcatImages(),
            DictToXY(),
        ]
    )

    ThreeDPW("train", protocol="train-test", transform=transform)

    train_data = ThreeDPW("train", protocol="train-test", transform=transform)
    validation_data = ThreeDPW("validation", protocol="train-test", transform=transform)
    test_data = ThreeDPW("test", protocol="train-test", transform=transform)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_dataloader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, validation_dataloader, test_dataloader


class Scale:

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample: dict):
        images, joint_position, width, height = (
            sample["images"],
            sample["joint_position"],
            sample["width"],
            sample["height"],
        )

        if isinstance(self.output_size, int):
            new_height = self.output_size
            new_width = self.output_size
        else:
            new_height, new_width = self.output_size

        new_height, new_width = int(new_height), int(new_width)

        for i, image in enumerate(images):
            if image:
                images[i] = image.resize((new_width, new_height))

        for i in range(len(joint_position)):
            if (i + 1) % 2 == 1:
                joint_position[i] = joint_position[i] / width
            elif (i + 1) % 2 == 0:
                joint_position[i] = joint_position[i] / height

        sample["images"] = images
        sample["joint_position"] = joint_position
        sample["width"] = new_width
        sample["height"] = new_height

        return sample


class Depth:

    def __init__(self):
        encoder = "vits"  # can also be 'vitb' or 'vitl'
        self.model = dpt.DepthAnything.from_pretrained(
            f"LiheYoung/depth_anything_{encoder}14"
        ).to(device)

    def __call__(self, sample: dict):

        images = sample["images"]

        for i, image in enumerate(images):
            if image is not None:
                depth = self.model(image.unsqueeze(0))
                images[i] = torch.cat((image, depth), dim=0)

        sample["images"] = images
        sample["channel_count"] += 1

        return sample


class OpticalFlow:

    def __init__(self):

        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(
            device
        )
        self.model = self.model.eval()

    def __call__(self, sample: dict):

        images = sample["images"]
        height, width = sample["height"], sample["width"]

        if images[0] is not None:
            optical_flow = torch.zeros((1, 2, height, width)).to(device)
            images[0] = torch.cat((images[0], optical_flow.squeeze(0)), dim=0)

        for i in range(1, len(images)):
            first_image = images[i - 1]
            second_image = images[i]
            if first_image is not None and second_image is not None:
                optical_flow = self.model(
                    first_image.unsqueeze(0)[:, :3, :, :],
                    second_image.unsqueeze(0)[:, :3, :, :],
                )
            elif second_image is not None:
                optical_flow = torch.zeros((1, 2, height, width)).to(device)
            else:
                continue

            if isinstance(optical_flow, list):
                optical_flow = optical_flow[-1]

            images[i] = torch.cat((second_image, optical_flow.squeeze(0)), dim=0)

        sample["images"] = images
        sample["channel_count"] += 2

        return sample

class NormalizeRGB:

    def __init__(self, std, mean):
        self.std = std
        self.mean = mean
        self.normalize = v2.Normalize(mean, std)

    def __call__(self, sample):

        images = sample["images"]

        for i, image in enumerate(images):
            if image is not None:
                image = torch.cat((self.normalize(image[:3, :, :]), image[3:, :, :]))
                images[i] = image

        sample["images"] = images

        return sample

class PILToTensor:
    """Convert PIL to Tensors."""

    def __call__(self, sample):

        images = sample["images"]

        for i, image in enumerate(images):
            if image:
                images[i] = functional.pil_to_tensor(image)
                images[i] = images[i].type(torch.float32)
                images[i] = images[i].to(device)

        sample["images"] = images

        return sample


class NoneToTensor:
    """Convert None to Tensors."""

    def __call__(self, sample):

        images = sample["images"]
        width = sample["width"]
        height = sample["height"]
        channel_count = sample["channel_count"]

        for i, image in enumerate(images):
            if image is None:
                images[i] = torch.zeros((channel_count, height, width)).to(device)

        sample["images"] = images

        return sample


class ConcatImages:

    def __call__(self, sample):

        images = sample["images"]
        final_image = images[0]

        for i in range(1, len(images)):
            final_image = torch.cat((final_image, images[i]), dim=0)

        del sample["images"]
        sample["final_image"] = final_image

        return sample


class DictToXY:

    def __call__(self, sample):

        X = sample["final_image"]
        y = torch.from_numpy(sample["joint_position"]).type(torch.float32)

        return (X, y)
