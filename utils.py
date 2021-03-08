import torch
import torch.nn as nn
from config import CHECKPOINT_DIR
import os

# TODO: figure out how to modularize this nicely
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
    char_map, idx_map = {}, {}
    for line in char_map_str.strip().split("\n"):
        c, num = line.split()
        char_map[c] = int(num)
        idx_map[int(num)] = c
    return char_map, idx_map

char_map, idx_map = create_char_map(char_map_str)

def text_to_target(text, char_map):
    target = []
    for c in text:
        if c == " ":
            target.append(char_map["<SPACE>"])
        else:
            target.append(char_map[c])
    return torch.Tensor(target)

def target_to_text(target):

    text = ""
    for idx in target:
        idx = idx.item()
        if idx == 1:
            text += " "
        else:
            text += idx_map[idx]
    return text

def weights_init_unif(module, a, b):
    for p in module.parameters():
        nn.init.uniform_(p.data, a=a, b=b)

def load_from_checkpoint(model, optimizer, checkpoint_name, device):

    path = os.join(CHECKPOINT_DIR, checkpoint_name)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss

def save_checkpoint(model, optimizer, epoch, activation, batch_size):

    filename = "activation-{}_batch-size-{}_epoch-{}.pt".format(activation, batch_size, epoch)
    save_path = CHECKPOINT_DIR + filename 

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, save_path)