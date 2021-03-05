import torch
import torch.nn as nn
import torchaudio
from config import hparams, DATASET_DIR
from preprocess import preprocess
from utils import weights_init_unif
from model import ASR_1
from training import train
from inference import test

def main():
    
    # todo: get cuda working
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    train_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="train-clean-100", download=True)
    dev_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="dev-clean", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(DATASET_DIR, url="test-clean", download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=preprocess, pin_memory=use_cuda)

    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=preprocess, pin_memory=use_cuda)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False, collate_fn=preprocess, pin_memory=use_cuda)

    # 1 channel input from feature spectrogram, 29 dim output from char_map + blank for CTC, 120 features
    net = ASR_1(in_dim=1, num_classes=29, num_features=120, activation="relu", dropout=0.3)
    net.to(device)
    weights_init_unif(net, hparams["weights_init_a"], hparams["weights_init_b"])

    # ADAM loss w/ lr=10e-4, batch size 20, initial weights initialized uniformly from [-0.05, 0.05], dropout w/ p=0.3 used in all layers except in and out
    # for fine tuning: SGD w/ lr 10e-5, l2 penalty w/ coeff=1e-5

    criterion = nn.CTCLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=hparams["ADAM_lr"])
    finetune_optimizer = torch.optim.SGD(net.parameters(), lr=hparams["SGD_lr"], weight_decay=hparams["SGD_l2_penalty"])

    # for epoch in range(1, hparams["epochs"] + 1):
    #     train(net, train_loader, criterion, optimizer, epoch, device)
        
    # TODO: Where/when to do dev set?

    test(net, test_loader, criterion, device)

if __name__ == "__main__":
    main()