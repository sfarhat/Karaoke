import torch
from utils import target_to_text

def test(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths in test_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            output = output.transpose(0, 1)
            loss = criterion(output, targets, input_lengths, target_lengths)
            
            # TODO: Decoding algo
            # Transpose back so that we can iterate over batch dimension
            output = output.transpose(0, 1)
            for log_probs in output:
                guessed_target = greedy_decode(torch.argmax(log_probs, dim=1))

def greedy_decode(char_indices):

    # TODO: incorporate target_length in here to reduce amount of work necessary
    transcript = []
    blank_seen = False
    prev = None
    for i in char_indices:
        if i == prev and not blank_seen:
           continue
        elif i == 0:
            blank_seen = True
        else:
            transcript.append(i)
            blank_seen = False

    return target_to_text(transcript)