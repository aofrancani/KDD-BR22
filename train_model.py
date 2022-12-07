import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import pickle
from dataloader import KDD22
from model import CNNVO


def val_model(model, val_loader, criterion):
    epoch_loss = 0
    with tqdm(val_loader, unit="batch") as tepoch:
        for images, altitude, delta, odom in tepoch:
            tepoch.set_description(f"Validating ")

            if torch.cuda.is_available():
                images, altitude, delta, odom = images.cuda(), altitude.cuda(), delta.cuda(), odom.cuda()
            altitude = altitude.unsqueeze(1)  # correct batch shape [bsize x 1]
            delta = delta.unsqueeze(1)

            # predict odom
            estimated_odom = model(images.float(), altitude.float(), delta.float())

            # compute loss
            loss = criterion(estimated_odom, odom.float())
            epoch_loss += loss.item()
            tepoch.set_postfix(val_loss=loss.item())

    return epoch_loss / len(val_loader)


def train_model(model, train_loader, criterion, optimizer, epoch, tensorboard_writer):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1

    with tqdm(train_loader, unit="batch") as tepoch:
        for images, altitude, delta, odom in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            if torch.cuda.is_available():
                images, altitude, delta, odom = images.cuda(), altitude.cuda(), delta.cuda(), odom.cuda()
            altitude = altitude.unsqueeze(1)  # correct batch shape [bsize x 1]
            delta = delta.unsqueeze(1)

            # predict odom
            estimated_odom = model(images.float(), altitude.float(), delta.float())

            # loss = criterion(estimated_odom, odom.float())
            loss = torch.sqrt(criterion(estimated_odom, odom.float()))

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

            # log tensorboard
            tensorboard_writer.add_scalar('training_loss', loss.item(), iter)
            iter += 1
    return epoch_loss / len(train_loader)


def train(model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args):
    best_val = args["best_val"]
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]

    for epoch in range(args["epoch_init"], epochs):
        # training for one epoch
        model.train()
        train_loss = train_model(model, train_loader, criterion, optimizer, epoch, tensorboard_writer)

        # validate model
        model.eval()
        with torch.no_grad():
            val_loss = val_model(model, val_loader, criterion)

        print(f"Epoch: {epoch} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} \n")

        # save best mode
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_val": best_val,
        }
        if val_loss < best_val:
            print(f"Saving new best model -- loss decreased from {best_val:.4f} to {val_loss:.4f} \n")
            best_val = val_loss
            state["best_val"] = best_val
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_best.pth"))

        # save model
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))

        # log loss in TensorBoard
        tensorboard_writer.add_scalar("train_loss", train_loss, epoch)
        tensorboard_writer.add_scalar("val_loss", val_loss, epoch)
    return


if __name__ == "__main__":

    # set hyperparameters and configuration
    args = {
        "data_dir": "dataset",
        "bsize": 4,  # batch size
        "lr": 1e-3,  # learning rate
        "momentum": 0.9,  # SGD momentum
        "weight_decay": 0.0005,  # SGD momentum
        "epoch": 200,  # train iters each timestep
        "epsilon": 0.001,  # linear decay of exploration policy
        "checkpoint_path": "ckpt/exp1",  # path to save checkpoint
        "checkpoint": None,  # checkpoint
    }

    # create ckpt_path and save args
    if not os.path.exists(args["checkpoint_path"]):
        os.makedirs(args["checkpoint_path"])

    with open(args["checkpoint_path"] + '/args.pkl', 'wb') as f:
        pickle.dump(args, f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tensorboard writer
    TensorBoardWriter = SummaryWriter(log_dir=args["checkpoint_path"])

    # preprocessing operation
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # train and val dataloader
    train_dataset = KDD22(args["data_dir"], split="train", transform=preprocess)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args["bsize"],
                                               shuffle=True,
                                               )

    val_dataset = KDD22(args["data_dir"], split="val", val_split=0.01, transform=preprocess)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             )

    # build and load model
    model = CNNVO()
    args["epoch_init"] = 1
    args["best_val"] = np.inf
    if args["checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]), map_location=device)
        args["epoch_init"] = checkpoint["epoch"] + 1
        args["best_val"] = checkpoint["best_val"]
        model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        model.cuda()

    # define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    # train network
    train(model, train_loader, val_loader, criterion, optimizer, TensorBoardWriter, args)
