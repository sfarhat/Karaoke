import torch
import torch.nn as nn
import torchaudio
from utils import text_to_target, char_map

def features_from_waveform(waveform):

    """
    Raw audio is transformed into 40-dimensional log mel-filter-bank (plus energy term) coefficients with deltas and delta-deltas, which reasults in 123 dimensional features.
    Each dimension is normalized to have zero mean and unit variance over the training set.
    Basically this is just MFCC but without taking DCT at the end, but for the sake of cleanliness, I'll stick with MFCC for now. 
    Also, I don't know what they mean by "energy term" (aren't the coefficients already energy terms?) so I'm omitting that.

    :param waveform: Time series data representing spoken input. Shape (channel, amplitude, time)
    :returns: "Spectrogram" of MFCC, delta, and delta-delta features. Shape (120, time)
    """

    # Waveform has channel first dimension, gives shape (1, ...) which causes shape problems when stacking features
    data = waveform.squeeze(dim=0)

    # Grab desired features
    mfcc_features = torchaudio.transforms.MFCC(log_mels=True)(data), # Takes in audio of dimension (..., time) returns (..., n_mfcc, time) where n_mfcc defaults to 40
    deltas = torchaudio.functional.compute_deltas(mfcc_features)
    delta_deltas = torchaudio.functional.compute_deltas(deltas)

    # Stack features together
    input_features = torch.cat((mfcc_features, deltas, delta_deltas), 0)

    # Normalize (0 mean, 1 std) features along time dimension
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

    This is fed into the DataLoader as the collate_fn, so keep in mind batch dimension.

    :param dataset: Batched raw waveforms with shape (batch_size, channel, amplitude, time)
    :returns inputs: Input "filter banks" to network (batch_size, num_channels, num_features, time). Padded along time dimension.
    :returns input_lengths: Python list of respective lengths (time) of inputs in batch 
    :returns targets: Transcript of sound converted to Tensor of integers instead of characters. Padded for uniform lengths within batch.
    :returns target_lengths: Python list of respective lenghts of transcripts in batch
    """

    # These are necessary for CTCLoss
    inputs = [] 
    targets = [] 
    # For these, CTCLoss expects them to be pre-padding
    input_lengths = [] 
    target_lengths = [] 

    # Each waveform has different lengths of time dimension, so we need to pad them. 
    # The built-in fn assumes trailing dimensions of all sequences are the same, so we have to transpose to pad the correct dimension since time (dim=1) varies, then transpose back after.

    for waveform, _, transcript, _, _, _ in dataset:
        # Output of features_from_waveform have dim: (120, time)
        features = features_from_waveform(waveform).transpose(0, 1)
        inputs.append(features)
        input_lengths.append(features.shape[0])

        target = text_to_target(transcript.lower(), char_map)
        targets.append(target)
        target_lengths.append(len(target))

    # Need to add back (unsqueeze) channel dimension
    # Returns dimensions: (batch, time, features), so transpose appropriately
    # Side note: this will pad all samples within the same batch (not overall) to be the same dimensions
    # This transformation doesn't affect the model architecture since we are only flattening/connecting the feature dimension, not the time one (nn.Linear allows for this flexibility), so we can have variable length inputs across batches in this regard
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).unsqueeze(1).transpose(2, 3) 

    # This will pad with 0, which represents the blank, but this shouldn't be a problem since we're providing the target_length of the unpadded target
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return inputs, input_lengths, targets, target_lengths

