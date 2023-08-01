import argparse
import torch
import pandas as pd
import numpy as np
np.set_printoptions(precision=3, linewidth=125)
torch.set_printoptions(precision=3, linewidth=125)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device", "\n")


# Get data -----------------------------------------------------------------------------------------------------------------

def get_data(filepath, date_filter):
    
    # CSV
    d = pd.read_csv(filepath, index_col=None)
    d['date_stamp'] = pd.to_datetime(d['date_stamp'], format="%d/%m/%Y")
    
    # Date filter
    min_date = date_filter
    max_date = d['date_stamp'].max()

    idx = d[(d['date_stamp'] > min_date) & (d['date_stamp'] < max_date)]['date_stamp'].values
    Y = d.query('(date_stamp > @min_date) & (date_stamp < @max_date)')['fwd_rtn'].values
    X = d[(d['date_stamp'] > min_date) & (d['date_stamp'] < max_date)].iloc[:, 2:4].values
    X = torch.from_numpy(X).to(torch.float32)

    return X, Y, idx


def dummy_data(n=20, mask=False):
    d = np.random.randint(low=2, high=5, size=(n,3))
    e = (np.random.rand(n) + 0 + d[:,0]**2)[:, None]
    e = np.concatenate((e, (np.random.rand(n) + 2 + d[:,0]**2)[:, None]), axis=1)
    e = np.concatenate((e, (np.random.rand(n) + 1 + np.sqrt(d[:,0]))[:, None]), axis=1)
    e = np.concatenate((e, (np.random.rand(n) + 2 + d[:,1]**2)[:, None]), axis=1)
    e = np.concatenate((e, (np.random.rand(n) + 1 + np.sqrt(d[:,1]))[:, None]), axis=1)
    e = np.concatenate((e, (np.random.rand(n) + 2 + d[:,2]**2)[:, None]), axis=1)
    e = np.concatenate((e, (np.random.rand(n) + 1 + np.sqrt(d[:,2]))[:, None]), axis=1)
    e = np.concatenate((e, (np.random.rand(n) + 2 + d[:,2]**.75)[:, None]), axis=1)
    e = np.concatenate((e, (np.random.rand(n) + 1 + np.sin(d[:,2]))[:, None]), axis=1)
    if mask:
        e = e * np.random.randint(low=0, high=2, size=(e.shape))
    e = torch.from_numpy(e).to(torch.float32)
    e_mean = torch.mean(e, dim=0)
    e_std = torch.std(e, dim=0)
    e = (e - e_mean) / e_std
    return d, e




# Define model -------------------------------------------------------------------------------------------------------------

class AutoEnc(torch.nn.Module):
    def __init__(self, layer_widths):
        super(AutoEnc, self).__init__()

        # Encoder
        self.encoder = torch.nn.ModuleList()
        encz = zip(layer_widths[:-1], layer_widths[1:])
        *hidden, last = encz
        for n_in, n_out in hidden:
            self.encoder.append(torch.nn.Linear(n_in, n_out))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(last[0], last[1]))
        self.encoder = torch.nn.Sequential(*self.encoder)

        # Decoder
        self.decoder = torch.nn.ModuleList()
        decz = zip(reversed(layer_widths[1:]), reversed(layer_widths[:-1]))
        *hidden, last = decz
        for n_in, n_out in hidden:
            self.decoder.append(torch.nn.Linear(n_in, n_out))
            self.decoder.append(torch.nn.ReLU())
        self.decoder.append(torch.nn.Linear(last[0], last[1]))
        self.decoder = torch.nn.Sequential(*self.decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

        


# Instantiate model & training loop ----------------------------------------------------------------------------------------

def train_loop(data, layer_width, epochs, verbose=True):

    layer_widths = [data.shape[1]] + layer_width

    # Instantiate model
    model = AutoEnc(layer_widths)
    print("Model architecture:", "\n", model, "\n")
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e8)


    # Training loop
    for epoch in range(epochs):
        encoded, decoded = model(data)
        loss = loss_fn(decoded, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print loss
        if epoch % 20 == 0 and verbose == True:
            print(f"Training error: {loss.item():>7f}")

    return encoded



# Main ---------------------------------------------------------------------------------------------------------------------

def main(args):
    
    # Convert date parameter from string to date, for use as data frame filter
    #date_filter = dt.datetime.strptime(args.date_filter, '%Y-%m-%d').date()
    #date_filter = args.date_filter

    # Training data
    X, _, _ = get_data(filepath=args.filepath, date_filter=args.date_filter)
    _, X = dummy_data()

    encoded = train_loop(data=X, layer_width=[7,5,3], epochs=200)

    encoded_np = encoded.detach().numpy()
    pd.DataFrame(encoded_np) \
        .to_csv('/c/Users/brent/Documents/R/Misc_scripts/m02_preds.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear regression")
    parser.add_argument("-d", "--date_filter", nargs="?", default="2022-12-31", type=str)
    parser.add_argument("-f", "--filepath"   , nargs="?", default="/c/Users/brent/Documents/R/Misc_scripts/stocks.csv", type=str)
    parser.add_argument("-a", "--data"       , nargs="?", default=False, type=bool)

    args = parser.parse_args()

    main(args)


# python ~/numpyro_models/numpyro_models/m02.py -d 2021-12-31
# python ~/numpyro_models/numpyro_models/m02.py -a True