import argparse
import torch
import pandas as pd
import numpy as np
np.set_printoptions(precision=3, linewidth=125)
torch.set_printoptions(precision=3, linewidth=125)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device", "\n")

# https://www.geeksforgeeks.org/how-to-pass-a-list-as-a-command-line-argument-with-argparse/
def list_of_ints(ints):
    return list(map(int, ints.split(',')))

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


def dummy_data(n=7, noise_scale=1, mask=False, scale=False, complex=False):
    n = n if n % 2 == 1 else n - 1
    d = np.linspace(21,15,n)[:, None]
    f0 = np.linspace(5,8,int(np.ceil(n*3/4)))
    f1 = f0[::-1] # reverse
    cut = n - len(f1)
    f1 = f1[1:,]  # all except first
    f1 = f1[0:3,]
    f = np.concatenate((f0,f1))
    d = np.concatenate((d, f[:, None]), axis=1)
    f0= np.linspace(5,8,int(np.ceil(n/2)))
    f1= f0[::-1] # reverse
    f1= f1[1:,]  # all except first
    f = np.concatenate((f0,f1))
    d = np.concatenate((d, f[:, None]), axis=1)
    if complex:
        e = (d[:,0]**2)[:, None]
        e = np.concatenate((e, (+2 + np.sqrt(d[:,0]))[:, None]), axis=1)
        e = np.concatenate((e, (-5 +         d[:,0]**2)[:, None]), axis=1)
        e = np.concatenate((e, (+1 + np.sqrt(d[:,1]))[:, None]), axis=1)
        e = np.concatenate((e, (+2 +         d[:,1]**2)[:, None]), axis=1)
        e = np.concatenate((e, (+1 + np.sqrt(d[:,2]))[:, None]), axis=1)
        e = np.concatenate((e, (+2 +         d[:,2]**.75)[:, None]), axis=1)
        e = np.concatenate((e, (+1 +  np.sin(d[:,2]))[:, None]), axis=1)
        e = e + np.random.normal(loc=0, scale=noise_scale, size=(e.shape))
    else:
        e = (d[:,0]*2)[:, None]
        e = np.concatenate((e, (d[:,0]*0.5)[:, None]), axis=1)
        e = np.concatenate((e, (d[:,0]*2  )[:, None]), axis=1)
        e = np.concatenate((e, (d[:,1]*0.5)[:, None]), axis=1)
        e = np.concatenate((e, (d[:,1]*2  )[:, None]), axis=1)
        e = np.concatenate((e, (d[:,2]*0.5)[:, None]), axis=1)
        e = np.concatenate((e, (d[:,2]*2  )[:, None]), axis=1)
        e = np.concatenate((e, (d[:,2]*2.9)[:, None]), axis=1)
        e = e + np.random.normal(loc=0, scale=noise_scale, size=(e.shape))
    if mask:
        e = e * np.random.randint(low=0, high=2, size=(e.shape))
    e = torch.from_numpy(e).to(torch.float32)
    if scale:
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
            self.encoder.append(torch.nn.Sigmoid())
        self.encoder.append(torch.nn.Linear(last[0], last[1]))
        self.encoder = torch.nn.Sequential(*self.encoder)

        # Decoder
        self.decoder = torch.nn.ModuleList()
        decz = zip(reversed(layer_widths[1:]), reversed(layer_widths[:-1]))
        *hidden, last = decz
        for n_in, n_out in hidden:
            self.decoder.append(torch.nn.Linear(n_in, n_out))
            self.decoder.append(torch.nn.Sigmoid())
        self.decoder.append(torch.nn.Linear(last[0], last[1]))
        self.decoder = torch.nn.Sequential(*self.decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

        


# Instantiate model & training loop ----------------------------------------------------------------------------------------

def train_loop(data, layer_width, epochs, learn_rate, verbose=True):

    layer_widths = [data.shape[1]] + layer_width

    # Instantiate model
    model = AutoEnc(layer_widths)
    print("Model architecture:", "\n", model, "\n")
    loss_fn = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e8)
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)


    # Training loop
    for epoch in range(epochs):
        encoded, decoded = model(data)
        loss = loss_fn(decoded, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print loss
        if epoch % 50 == 0 and verbose == True:
            print(f"Training error: {loss.item():>7f}")

    return encoded, decoded



# Main ---------------------------------------------------------------------------------------------------------------------

def main(args):
    
    # Convert date parameter from string to date, for use as data frame filter
    #date_filter = dt.datetime.strptime(args.date_filter, '%Y-%m-%d').date()
    #date_filter = args.date_filter

    # Training data
    #X, _, _ = get_data(filepath=args.filepath, date_filter=args.date_filter)
    _, X = dummy_data(n=7, scale=False, complex=True)
    
    # Standardise
    X_mean = torch.mean(X, dim=0)
    X_std = torch.std(X, dim=0)
    X_norm = (X - X_mean) / X_std

    #encoded, decoded = train_loop(data=X_norm, layer_width=args.layer_width, epochs=args.epochs, learn_rate=args.learn_rate)
    enc_norm, dec_norm = train_loop(data=X_norm, layer_width=args.layer_width, epochs=args.epochs, learn_rate=args.learn_rate)
    encoded = enc_norm #(enc_norm * X_std) + X_mean
    decoded = (dec_norm * X_std) + X_mean

    pd.DataFrame(encoded.detach().numpy()) \
        .to_csv('/c/Users/brent/Documents/R/Misc_scripts/m02_preds.csv')
    
    print(np.sqrt(((X.detach().numpy() - decoded.detach().numpy()) ** 2).mean()))    # rmse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder")
    parser.add_argument("-d", "--date_filter", nargs="?", default="2022-12-31", type=str)
    parser.add_argument("-f", "--filepath"   , nargs="?", default="/c/Users/brent/Documents/R/Misc_scripts/stocks.csv", type=str)
    parser.add_argument("-l", "--learn_rate" , nargs="?", default=1e-1, type=float)
    parser.add_argument("-e", "--epochs"     , nargs="?", default=100, type=int)
    parser.add_argument("-y", "--layer_width", nargs="?", default=[5,3], type=list_of_ints)

    args = parser.parse_args()

    main(args)


# conda activate pytorch_pyro
# python ~/numpyro_models/numpyro_models/m02.py -d 2021-12-31
# python ~/numpyro_models/numpyro_models/m02.py -l 1e-2 -e 500 -y 5,5,3