import torch

def test(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths in test_loader:

            output = model(inputs)

            loss = criterion(output, targets, input_lengths, target_lengths)
            
            # TODO: Decoding algo