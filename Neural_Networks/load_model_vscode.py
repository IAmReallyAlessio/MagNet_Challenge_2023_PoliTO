import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import math
import matplotlib.pyplot as plt
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define a fully connected layers model with three inputs (frequency, flux density, duty ratio) and one output (core loss).
        self.layers = nn.Sequential(
            nn.Linear(4, 15),
            nn.ReLU(),
            nn.Linear(15, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
        )

    def forward(self, x):
        return self.layers(x)

def set_input(freq, flux, duty, temp):

    Freq_t = np.log10(freq)
    Flux_t = np.log10(flux)
    Duty_t = np.array(duty)
    Temperature_t = np.array(temp)

    tmp = np.array([Freq_t, Flux_t, Duty_t, Temperature_t])

    in_tensor = torch.from_numpy(tmp).view(1, 4)

    return in_tensor

def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cpu"}
    modello = Net().double().to(device)

    modello.load_state_dict(torch.load("C:\\Users\\fabio\\Desktop\\MagNet\\models\\Model_FNN_N87.sd"))

    NUM_EPOCH = 1000
    BATCH_SIZE = 64
    DECAY_EPOCH = 100
    DECAY_RATIO = 0.5
    LR_INI = 0.02



    testing = set_input(354610, 0.05669549996007316, 0.6982421875, 90)

    test_loader = torch.utils.data.DataLoader(testing, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(modello.parameters(), lr=LR_INI)

    risultato = modello(testing.to(device))
    risultatone = 10**(risultato.detach().cpu().numpy())
    print(risultatone[0][0])

if __name__ == "__main__":
    main()