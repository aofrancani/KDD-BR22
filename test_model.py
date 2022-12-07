import torch
from torchvision import transforms
from model import CNNVO
from dataloader import KDD22
import os
from tqdm import tqdm
import pandas as pd


# set hyperparameters and configuration
split = "test"
checkpoint_path = "ckpt/exp1/"   # path to saved checkpoint
checkpoint = "checkpoint_best.pth"                       # name of checkpoint
submission_path = checkpoint_path   # path to save submission file

device = "cuda" if torch.cuda.is_available() else "cpu"

# preprocessing operation
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# test dataloader
test_dataset = KDD22(data_dir="dataset", split=split, transform=preprocess)
test_data = test_dataset.data
dataloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         )

# load model
model = CNNVO()
if checkpoint is not None:
    checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint), map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
if torch.cuda.is_available():
    model.cuda()


# evaluate model
model.eval()
with torch.no_grad():
    for images, altitude, delta, odom in tqdm(dataloader, unit="sample"):
        if torch.cuda.is_available():
            images, altitude, delta, odom = images.cuda(), altitude.cuda(), delta.cuda(), odom.cuda()
        altitude = altitude.unsqueeze(1)  # correct batch shape [bsize x 1]
        delta = delta.unsqueeze(1)

        # predict odom
        pred_odom = model(images.float(), altitude.float(), delta.float())

        # update odom in dataloader.dataset.data
        filename = dataloader.dataset.filename

        # update test dataframe
        test_dataset.data.loc[test_data["Filename"] == filename, "North"] = pred_odom[0][0].cpu().numpy()
        test_dataset.data.loc[test_data["Filename"] == filename, "East"] = pred_odom[0][1].cpu().numpy()

# generate submission (df --> .csv)
new_rows = []
for idx, row in test_data.iterrows():
    row_north = f'{row["Filename"]}:North'
    row_east = f'{row["Filename"]}:East'
    new_rows.append([row_north, row['North']])
    new_rows.append([row_east, row['East']])
df_submission = pd.DataFrame(new_rows, columns=['Id', 'Predicted'])
df_submission.to_csv(os.path.join(submission_path, "submission_file.csv"), sep=',', index=False)