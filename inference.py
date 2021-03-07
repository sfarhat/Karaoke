import torch
from utils import target_to_text

def test(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths in test_loader:

            output = model(inputs)

            loss = criterion(output, targets, input_lengths, target_lengths)
            
            # TODO: Decoding algo
            for log_probs in output:
                gussed_target = greedy_decode(torch.argmax(log_probs, dim=1))

def greedy_decode(log_probs):

    transcript = []
    blank_seen = False
    prev = None
    for i in log_probs:
        if i == prev and not blank_seen:
           continue
        elif i == 0:
            blank_seen = True
        else:
            transcript.append(i)
            blank_seen = False

    return target_to_text(transcript)