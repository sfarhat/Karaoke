import torch
import torch.nn as nn
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from config import hparams, DATASET_DIR
from preprocess import preprocess
from utils import weights_init_unif, load_from_checkpoint, save_checkpoint
from model import ASR_1
from training import train
from inference import test
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    train_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="train-clean-100", download=True)
    # dev_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="dev-clean", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="test-clean", download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=preprocess, pin_memory=use_cuda)

    # dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=preprocess, pin_memory=use_cuda)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False, collate_fn=preprocess, pin_memory=use_cuda)

    # 1 channel input from feature spectrogram, 29 dim output from char_map + blank for CTC, 120 features
    net = ASR_1(in_dim=1, num_classes=29, num_features=120, activation=hparams["activation"], dropout=0.3)
    net.to(device)
    weights_init_unif(net, hparams["weights_init_a"], hparams["weights_init_b"])

    # ADAM loss w/ lr=10e-4, batch size 20, initial weights initialized uniformly from [-0.05, 0.05], dropout w/ p=0.3 used in all layers except in and out
    # for fine tuning: SGD w/ lr 10e-5, l2 penalty w/ coeff=1e-5

    criterion = nn.CTCLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=hparams["ADAM_lr"])
    finetune_optimizer = torch.optim.SGD(net.parameters(), lr=hparams["SGD_lr"], weight_decay=hparams["SGD_l2_penalty"])

    writer = SummaryWriter()
    # writer.add_graph(net)

    for epoch in range(hparams["epochs"]):
        train(net, train_loader, criterion, optimizer, epoch, device, writer)
        save_checkpoint(net, optimizer, epoch, hparams["activation"], hparams["batch_size"])
    
    # TODO: Where/when to do dev set?

    # net, _, _, _ = load_from_checkpoint(net, optimizer, "activation-relu_batch-size-3_epoch-3.pt", device)

    # pass
    # filter_list = list(net.children())[0][0].conv.weight.data
    # im = filter_list[0]
    # im = np.transpose(im, (1,2,0))
    # plt.imshow(im)
    # plt.show()

    test(net, test_loader, criterion, device)
    

if __name__ == "__main__":
    main()