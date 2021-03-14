import torch
from config import CHECKPOINT_DIR
from utils import save_checkpoint

def train(model, train_loader, criterion, optimizer, epoch, device, writer):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_num, data in enumerate(train_loader): 
        inputs, input_lengths, targets, target_lengths = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # this is shape (batch size, time, num_classes)
        output = model(inputs)

        # output passed in should be of shape (time, batch size, num_classes)
        output = output.transpose(0, 1)
        loss = criterion(output, targets, input_lengths, target_lengths)
        loss.backward()

        optimizer.step() 

        if batch_num % 100 == 0 or batch_num == data_len:
            writer.add_scalar("Loss/train", loss.item(), epoch * data_len + batch_num)
            writer.flush()     
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_num * len(inputs), data_len,
                100. * batch_num / len(train_loader), loss.item()))