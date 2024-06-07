"""_summary_."""

import pathlib
import pickle

import data_setup
import engine
import torch

from vision_mamba.model import Vim

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-3

train_dataloader, validation_dataloader, test_dataloader = (
    data_setup.get_dataloaders(batch_size=BATCH_SIZE)
)

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

loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

result = engine.train(
    model=model,
    model_name="vim",
    train_dataloader=train_dataloader,
    validation_dataloader=validation_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device,
)

with open(
    pathlib.Path(__file__).parent.parent.absolute() / "metrics" / "vim.pkl", "wb"
) as file:
    pickle.dump(result, file)

# torch.save(
#     labels[0], pathlib.Path(__file__).parent.parent.absolute() / "results" / "vim.pt"
# )
# torch.save(
#     labels[1], pathlib.Path(__file__).parent.parent.absolute() / "results" / "real.pt"
# )
