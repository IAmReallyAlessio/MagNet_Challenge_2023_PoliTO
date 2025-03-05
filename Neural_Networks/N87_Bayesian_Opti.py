import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import time
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization

class Net(nn.Module):
    def __init__(self, l1, l2, l3):
        super(Net, self).__init__()
        # Define a fully connected layers model with three inputs (frequency, flux density, duty ratio) and one output (core loss).
        self.layers = nn.Sequential(
            nn.Linear(4, l1),
            nn.Tanh(),
            nn.Linear(l1, l2),
            nn.Tanh(),
            nn.Linear(l2, l3),
            nn.Tanh(),
            nn.Linear(l3, 1),
        )

    def forward(self, x):
        return self.layers(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





def get_dataset():
    # Load .json Files
    with open('Dataset_tri_N87_mod.json','r') as load_f:
        DATA = json.load(load_f)

    Freq = DATA['Frequency']
    Flux = DATA['Flux_Density']
    Duty = DATA['Duty_Ratio']
    Temperature = DATA['Temperature']
    Power = DATA['Power_Loss']

    # Compute labels
    # There's approximalely an exponential relationship between Loss-Freq and Loss-Flux.
    # Using logarithm may help to improve the training.
    Freq = np.log10(Freq)
    Flux = np.log10(Flux)
    Duty = np.array(Duty)
    Temperature = np.array(Temperature)
    Power = np.log10(Power)

    # Reshape data
    Freq = Freq.reshape((-1,1))
    Flux = Flux.reshape((-1,1))
    Duty = Duty.reshape((-1,1))
    Temperature = Temperature.reshape((-1,1))

    """ print(np.shape(Freq))
    print(np.shape(Flux))
    print(np.shape(Duty))
    print(np.shape(Temperature))
    print(np.shape(Power)) """

    temp = np.concatenate((Freq, Flux, Duty, Temperature),axis=1)

    in_tensors = torch.from_numpy(temp).view(-1, 4)
    out_tensors = torch.from_numpy(Power).view(-1, 1)

    # # Save dataset for future use
    # np.save("dataset.fc.in.npy", in_tensors.numpy())
    # np.save("dataset.fc.out.npy", out_tensors.numpy())

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


def _try_(l1, l2, l3):
    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    NUM_EPOCH = 200
    BATCH_SIZE = 128
    DECAY_EPOCH = 100
    DECAY_RATIO = 0.5
    LR_INI = 0.001

    # Select GPU as default device
    device = torch.device("cuda")

    # Get dataset
    dataset = get_dataset()

    splits = torch.utils.data.random_split(dataset, [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2])

    # Split the dataset
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}

    test_loader = torch.utils.data.DataLoader(splits[8], batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_size = len(splits[8])

    # Setup network
    net = Net(l1,l2, l3).double().to(device)

    # Log the number of parameters
    #print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI)

    # Setup values for plot of loss function
    y_costf = []
    y_validf = []
    epochs = []


    # Train the network    
    #print("Starting Training:")

    for epoch_i in range(NUM_EPOCH):
        rnd = random.randint(0,7)
        valid_loader = torch.utils.data.DataLoader(splits[rnd], batch_size=BATCH_SIZE, shuffle=False, **kwargs)
        valid_size = len(splits[rnd])
        train_size = 0
        train_loader = []
        for i in range(0,8):
            if i!=rnd:
                train_loader = torch.utils.data.ConcatDataset([train_loader, splits[i]])
                train_size += len(splits[i])

        train_loader = torch.utils.data.DataLoader(train_loader, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Compute Validation Loss
        with torch.no_grad():
            epoch_valid_loss = 0
            for inputs, labels in valid_loader:
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))

                epoch_valid_loss += loss.item()

        # Progress Bar
        if (epoch_i+1)%10 == 0:
          #print(".", end="", flush=True) 
          epochs.append(epoch_i)
          y_costf.append(epoch_train_loss / train_size * 1e5)
          y_validf.append(epoch_valid_loss / valid_size * 1e5)

    # Save the model parameters
    #torch.save(net.state_dict(), "/content/Model_FNN.sd")
    #print("Training finished! Model is saved!")

    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            y_pred.append(net(inputs.to(device)))
            y_meas.append(labels.to(device))

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    test_loss = F.mse_loss(y_meas, y_pred).item() / test_size * 1e5
#    print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / test_size * 1e5:.5f}")

    yy_pred = 10**(y_pred.cpu().numpy())
    yy_meas = 10**(y_meas.cpu().numpy())

    # Relative Error
    Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100
    Error_re_avg = np.mean(Error_re)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)
#    print(f"Relative Error: {Error_re_avg:.8f}")
#    print(f"RMS Error: {Error_re_rms:.8f}")
#    print(f"MAX Error: {Error_re_max:.8f}")

    #Save plot of loss functions
    plt.figure(figsize=(15,10))
    plt.plot(epochs,y_costf)
    plt.plot(epochs,y_validf)
    plt.legend(["Train_Loss", "Valid_Loss"])
    plt.grid()
    plt.savefig(f".\\Bayes_opti_N87\\tentativo_{l1}_{l2}_{l3}.png")
    plt.close()
    
    return test_loss


def wrapper(l1, l2, l3):
    l1 = round(l1)
    l2 = round(l2)
    l3 = round(l3)
    result = _try_(l1, l2, l3)
    return 1/result
        

def main():
    bo = BayesianOptimization(
        f = wrapper,
        pbounds = {"l1": (2,20), "l2": (2,20), "l3": (2,20)}
    )
    bo.maximize(init_points = 40, n_iter = 50)

    print("Final result:", bo.max)

if __name__ == "__main__":
    main()
