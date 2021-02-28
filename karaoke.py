import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# TODO: Get CUDA working
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


def get_libri_speech_dataset(dataset_dir: str, dataset: str="train-clean-100") -> torch.utils.data.Dataset:

    """
    Function to download LibriSpeech dataset.

    Inputs: 
    dataset_dir -- Path to directory where dataset should be located/downloaded
    dataset -- Type of dataset desired. Options are "train-clean-100", "train-clean-360", "train-clean-500", "dev-clean", "dev-other", "test-clean", "test-other"
    dataset

    Output: torch.utils.data.Dataset of tuples with contents (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
    """

    # can use either key or url for "url" parameter of dataset download function
    return torchaudio.datasets.LIBRISPEECH(dataset_dir, url=dataset, download=True)


char_map_str = """
 <SPACE> 1
 a 2
 b 3
 c 4
 d 5
 e 6
 f 7
 g 8
 h 9
 i 10
 j 11
 k 12
 l 13
 m 14
 n 15
 o 16
 p 17
 q 18
 r 19
 s 20
 t 21
 u 22
 v 23
 w 24
 x 25
 y 26
 z 27
 ' 28
 """

def create_char_map(char_map_str):
    char_map = {}
    for line in char_map_str.strip().split("\n"):
        c, num = line.split()
        char_map[c] = int(num)
    return char_map

char_map = create_char_map(char_map_str)

def text_to_target(text, char_map):
    target = []
    for c in text:
        if c == " ":
            target.append(char_map["<SPACE>"])
        else:
            target.append(char_map[c])
    return torch.Tensor(target)


# %%
def features_from_waveform(waveform):

    """
    Raw audio is transformed into 40-dimensional log mel-filter-bank (plus energy term) coefficients with deltas and delta-deltas, which reasults in 123 dimensional features. Each dimension is normalized to have zero mean and unit variance over the training set.

    Basically this is just MFCC but without taking DCT at the end, but for the sake of cleanliness, I'll stick with MFCC for now. Also, I don't know what they mean by "energy term" (aren't the coefficients already energy terms?) so I'm omitting that.
    """

    data = waveform.squeeze(dim=0)

    mfcc_features = torchaudio.transforms.MFCC(log_mels=True)(data)
    deltas = torchaudio.functional.compute_deltas(mfcc_features)
    delta_deltas = torchaudio.functional.compute_deltas(deltas)

    input_features = torch.cat((mfcc_features, deltas, delta_deltas), 0)

    input_features_normalized = nn.LayerNorm(input_features.shape[1], elementwise_affine=False)(input_features)

    return input_features_normalized

def preprocess(dataset):

    """
    Preprocesses dataset

    1. Convert waveforms to input features
    2. Convert transcripts to output class indices
    3. Get input sequence (feature) lengths before padding
    4. Get target lengths before padding
    5. Pad inputs and targets for consistent sizes
    """

    inputs = [] # Features that are input to model
    targets = [] # Target transcripts, NOT output of model, used for CTC Loss
    # For these, CTCLoss expects them to be pre-padding
    input_lengths = [] # Time lengths of input features
    target_lengths = [] # Length of target transcripts

    # Each waveform has different lengths of time dimension, so we need to pad them. The built-in fn assumes trailing dimensions of all sequences are the same, so we have to transpose to pad the correct dimension since time dim (dim=1) is different, then transpose back after.

    for waveform, _, transcript, _, _, _ in dataset:
        # Output of features_from_waveform have dim: 120 x time
        features = features_from_waveform(waveform).transpose(0, 1)
        inputs.append(features)
        input_lengths.append(features.shape[0])

        target = text_to_target(transcript.lower(), char_map)
        targets.append(target)
        target_lengths.append(len(target))

    # Returns dimensions: num_inputs (batch) x time x features, so transpose appropriately
    # Need to add back (unsqueeze) channel dimension
    # Side note: this will pad all samples within the same batch (not overall) to be the same dimensions
    # TODO: Think about how this affects the dimensions of conv -> fc layers
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).unsqueeze(1).transpose(2, 3) 

    # This will pad with 0, which represents the blank, but this shouldn't be a problem since we're providing the target_length of the unpadded target
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return inputs, input_lengths, targets, target_lengths


def maxout(x):
    # TODO: Implement this
    return x

class Conv_Maxout_Layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=(3,5), pool=False, pool_dim=(3,1), dropout=0.3):
        super(Conv_Maxout_Layer, self).__init__()

        # Use padding so that feature maps don't decrease in size throughout network. Makes things wrt dimensions nice when moving to FC layers.
        # Math says that we should pad by (1, 2) or equivalently (kernel_size[0]//2, kernel_size[1]//2) to keep dimensions the same
        # However, Paper says they only pad along frame/time axis, so they assume dimensions of features will decrease? good for temporal nature of output we want, but more math for me! 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=(0,2))
        # Using Dropout module vs functional will automatically disable it during model.eval()
        self.dropout = nn.Dropout(p=dropout)
        self.pool = pool
        self.pool_dim = pool_dim

    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = nn.MaxPool2d(kernel_size=self.pool_dim)(x)
        x = maxout(x)
        # doing this after pooling has faster runtime because less operations
        x = self.dropout(x)
        return x

class FC_Layer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.3):
        super(FC_Layer, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc(x)
        x = maxout(x)
        x = self.dropout(x)
        return x

class ASR_1(nn.Module):

        """
        Unlike the other layers, the first convolutional layer is followed by a pooling layer, which is described in section 2. The pooling size is 3 × 1, which mean swe only pool over the frequency axis. The filter size is 3 × 5 across the layers. The model has 128 feature maps in the first four convolutional layers and 256 feature maps in the remaining six convolutional layers. Each fully-connected layer has 1024 units. Maxout with 2 piece-wise linear functions is used as the activation function.
        """ 

        def __init__(self, in_dim, out_dim, num_features):
            super(ASR_1, self).__init__()
            self.in_dim = in_dim 
            self.cnn_layers = nn.Sequential(
                            Conv_Maxout_Layer(self.in_dim, 128, pool=True, dropout=0),
                            Conv_Maxout_Layer(128, 128),
                            Conv_Maxout_Layer(128, 128),
                            Conv_Maxout_Layer(128, 128),
                            Conv_Maxout_Layer(128, 256),
                            Conv_Maxout_Layer(256, 256),
                            Conv_Maxout_Layer(256, 256),
                            Conv_Maxout_Layer(256, 256),
                            Conv_Maxout_Layer(256, 256),
                            Conv_Maxout_Layer(256, 256),
            )

            # For feature maps, Conv2d layers keep time dimension the same, 3x5 conv decreases frequency dim by 2
            #                   MaxPool2d layer only along frequency dimension, 3x1 pooling decreases frequency dim by 2
            # Features dimenions of resulting feature map: num_features - 2 (conv1) - 2 (maxpool) - 2 (conv[2-10]) * 9 = num_features - 22
            # So in our use case, first fc layer should have dimensions = 256 * (120 - 22) = 25088
            # TODO: would be nice to automate this math. would require passing in number of certain kinds of layers and their underlying details (not super important)

            self.fc_layers = nn.Sequential(
                            FC_Layer(256 * (num_features - 22), 1024),
                            FC_Layer(1024, 1024),
                            FC_Layer(1024, 1024),
                            FC_Layer(1024, out_dim, dropout=0)
            )

        def forward(self, x):
            x = self.cnn_layers(x)
            # cnn output shape (batch_size, num_channels, num_features, time)
            x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]) # flattens channels and feature dimensions into one
            x = x.transpose(1, 2) # (batch_size, time, features)

            # Input: (N, *, H_{in}) where * means any number of additional dimensions and H_in=in_features
            x = self.fc_layers(x)


            # TODO: Add log softmax at end for CTCLoss to work, look into what dimension we should do this over
            # x = torch.nn.functional.log_softmax(x, dim=?)
            return x


def weights_init_unif(module, a, b):
    for p in module.parameters():
        nn.init.uniform_(p.data, a=a, b=b)


# %%
PATH_TO_DATASET_DIR = "/mnt/d/Datasets/"

# def main():

hparams = {
    "ADAM_lr": 10e-4,
    "batch_size": 20,
    "SGD_lr": 10e-5,
    "SGD_l2_penalty": 1e-5,
    "weights_init_a": -0.05,
    "weights_init_b": 0.05,
    "epochs": 10
}

train_dataset = get_libri_speech_dataset(PATH_TO_DATASET_DIR)
dev_dataset = get_libri_speech_dataset(PATH_TO_DATASET_DIR, "dev-clean")
test_dataset = get_libri_speech_dataset(PATH_TO_DATASET_DIR, "test-clean")

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=preprocess)

# dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=preprocess)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False, collate_fn=preprocess)

# 1 channel input from feature spectrogram, 29 dim output from char_map + blank for CTC, 120 features
net = ASR_1(1, 29, 120)
weights_init_unif(net, hparams["weights_init_a"], hparams["weights_init_b"])

# ADAM loss w/ lr=10e-4, batch size 20, initial weights initialized uniformly from [-0.05, 0.05], dropout w/ p=0.3 used in all layers except in and out
# for fine tuning: SGD w/ lr 10e-5, l2 penalty w/ coeff=1e-5

criterion = nn.CTCLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=hparams["ADAM_lr"])
finetune_optimizer = torch.optim.SGD(net.parameters(), lr=hparams["SGD_lr"], weight_decay=hparams["SGD_l2_penalty"])

# for epoch in range(hparams["epochs"]):
#     train(net, train_loader, criterion, optimizer)
    
# TODO: Where/when to do dev set?

# test(net, test_loader, criterion)


# %%
def train(model, train_dataset, criterion, optimizer):
    model.train()
    for inputs, input_lengths, targets, target_lengths in train_dataset:

        optimizer.zero_grad()

        # check what shape this is
        output = model(inputs)

        # output passed in should be of shape (time, batch size, num_classes)
        loss = criterion(output, targets, input_lengths, target_lengths)
        loss.backwards()

        optimizer.step() 

def test(model, test_dataset, criterion):
    model.eval()
    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths in test_dataset:

            output = model(inputs)

            loss = criterion(output, targets, input_lengths, target_lengths)
            
            # TODO: Decoding algo


# %%
for inputs, input_lengths, targets, target_lengths in test_loader:
    net(inputs)


# %%



